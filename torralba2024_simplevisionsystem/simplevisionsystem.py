import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageFilter
from matplotlib.colors import rgb_to_hsv
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from scipy.ndimage import label, grey_dilation, binary_dilation, convolve, distance_transform_edt

# hand-tuned input parameters
input_path = 'outputs/blockworld_downloaded.jpeg'
downsample_factor = 2 # downsampling factor
edge_thresh = 5 # edge magnitude threshold for edge detection
sat_thresh = 0.2 # saturation threshold for figure-ground segmentation
min_face_size = 60 # arbitrary threshold to drop small regions in the face segmentation
face_ori_rat = 0.3 # vert vs horiz rate for identifying 'face' orientation
theta_cam = np.deg2rad(15.0) # camera angle
ground_below = 8 # contact edge identification

# hand-tuned weights for the different constraints
w_ground   = 10.0
w_contact  = 8.0 
w_rise     = 10.0
w_perp     = 30.0
w_planar   = 1.0
w_flat     = 20.0
w_connect  = 25.0
w_occlude  = 10.0 

img = Image.open(input_path)

# 1. downsample image and convert to grayscale 
scale_factor=downsample_factor
blur_radius=scale_factor/2

blurred_img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
new_width = int(img.width / scale_factor)
new_height = int(img.height / scale_factor)
downsampled_img = blurred_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
downsampled_img.save("outputs/blockworlddownsampled.jpeg")

img_array = np.array(downsampled_img)
img_array_gray = img_array @ np.array([0.2989, 0.5870, 0.1140])

# 2. identify image edges, gradient magnitude and orientation, figure ground segmentation, and plot 
edge_kernel_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

gx = convolve(img_array_gray.astype(float), edge_kernel_x, mode='reflect') # convolve with reflection padding
gy = convolve(img_array_gray.astype(float), edge_kernel_x.T, mode='reflect') 

grad_mag = np.sqrt(gx ** 2 + gy ** 2)
grad_ori = np.arctan2(gy, gx)

edge_mask = grad_mag > edge_thresh # is there an edge at this pixel? 

ys, xs = np.where(edge_mask)
step = 8
sel = slice(None, None, step)
ys, xs = ys[sel], xs[sel]

grad_mag_downsampled = grad_mag[ys, xs] # mag of the graident at the selected edge pixels (downsampled)
grad_ori_downsampled = grad_ori[ys, xs] # orientation of the gradient at the selected edge pixels (downsampled)

# figure ground segmentation 
hsv = rgb_to_hsv(img_array / 255.0)
is_object = hsv[..., 1] > sat_thresh
is_object = convolve(is_object.astype(float), np.ones((3, 3)), mode='constant') > 0 # smooth the is object mask
is_ground = ~is_object

# horizontal vertical edge classifier 
tol = np.deg2rad(35)
dev = np.minimum(np.abs(grad_ori), np.pi - np.abs(grad_ori)) # distance of grad_ori from {0, +pi, -pi}
is_vertical = dev <= tol
is_horizontal = ~is_vertical

# lazy flood fill 'object face' segmentation 
edge_smooth = convolve(edge_mask.astype(float), np.ones((3, 3)), mode='constant') > 0 # smooth the edge map out to make them complete
face_region = is_object & ~edge_smooth

face_labels, n = label(face_region)

sizes = np.bincount(face_labels.ravel())
keep = np.where(sizes >= min_face_size)[0] # 
keep = keep[keep != 0]

face_id = np.zeros_like(face_labels)
for new, old in enumerate(keep, start=1):
    face_id[face_labels == old] = new

# fill every object pixel with the nearest labeled face (bc we weren't labeling the edge pixels before)
_, (ri, ci) = distance_transform_edt(face_id == 0, return_indices=True)
face_aug = face_id[ri, ci] * is_object

# label each face as vertical or horizontal based on its edge pixels --- if it has some vertical edges, it's vertical; else horizontal
is_horizontal_edge = is_horizontal & edge_mask
is_vertical_edge = is_vertical & edge_mask

is_horizontal_edge = convolve(is_horizontal_edge.astype(float), np.ones((5, 5)), mode='constant') > 0 # smooth the edge map out to make them complete
is_vertical_edge = convolve(is_vertical_edge.astype(float), np.ones((5, 5)), mode='constant') > 0 # smooth the edge map out to make them complete

is_vertical_face = np.zeros_like(face_id, dtype=bool)
is_horizontal_face = np.zeros_like(face_id, dtype=bool)
for id in np.unique(face_id):
    if id != 0:
        mask = face_id == id
        vert_edges = np.sum(is_vertical_edge[mask])
        horiz_edges = np.sum(is_horizontal_edge[mask])

        if vert_edges/horiz_edges > face_ori_rat:
            is_vertical_face[mask] = True
            is_horizontal_face[mask] = False
        else:
            is_vertical_face[mask] = False
            is_horizontal_face[mask] = True

ftype = {f: ('H' if is_horizontal_face[face_id == f].any() else 'V') for f in np.unique(face_id) if f != 0} # easy lookup for labels

# classiy edges as vertical occluding (vertical & occludes ground), horizontal occulding (horizontal & occludes ground), crease (interior face-face border)
# ignoring the figure/ground edges for now
near_bg = binary_dilation(is_ground, np.ones((5, 5)))

vert_band  = is_vertical_edge   & is_object
horiz_band = is_horizontal_edge & is_object

vert_occ   = vert_band  & near_bg
horiz_occ  = horiz_band & near_bg 
crease     = (edge_smooth & is_object) & ~near_bg  

# helpers for setting up and storing constraints 
H, W = grad_ori.shape
N = H * W

def idx(r, c): return r * W + c

rows, cols, vals, b = [], [], [], []
eq = 0
def add(coeffs, target, w=1.0):
    global eq
    for col, v in coeffs:
        rows.append(eq); cols.append(col); vals.append(w * v)
    b.append(w * target); eq += 1

cos_t, sin_t = np.cos(theta_cam), np.sin(theta_cam)

# constraint 1: ground -> Y = 0
for r, c in zip(*np.where(is_ground)):
    add([(idx(r, c), 1.0)], 0.0, w_ground)


# constraint 2: contact edges, i.e., object edges with ground below them -> Y = 0
for c in range(W):
    col = is_object[:, c]
    trans = np.where(col[:-1] & ~col[1:])[0]   
    if len(trans) == 0:
        continue
    r = trans.max()                              
    below = col[r+1 : r+1+ground_below]
    if (~below).all() and len(below) == ground_below:   
        add([(idx(r, c), 1.0)], 0.0, w_contact)

# constraint 3-6: if horizontal face, height (Y) is shared across full extent
# if vertical face, Y increases as you move up in image, d^2Y/dy^2 = 0 (perpindicular to floor), d^2Y/dx^2 = 0 (no horiz curves)  
on_occ = vert_occ | horiz_occ
for f in np.unique(face_aug):
    if f == 0:
        continue
    m = face_aug == f
    rr, cc = np.where(m)

    if ftype.get(f, 'H') == 'H':
        ref = idx(rr[0], cc[0])
        for r, c in zip(rr[1:], cc[1:]):
            wf = w_occlude if on_occ[r, c] else w_flat
            add([(idx(r, c), 1.0), (ref, -1.0)], 0.0, wf)
    else:
        for r, c in zip(rr, cc):
            wr = w_occlude if on_occ[r, c] else w_rise
            if r - 1 >= 0 and m[r - 1, c]:
                add([(idx(r - 1, c), 1.0), (idx(r, c), -1.0)], 1.0 / cos_t, wr)
            if 0 < r < H - 1 and m[r - 1, c] and m[r + 1, c]:
                add([(idx(r-1,c),1.0),(idx(r,c),-2.0),(idx(r+1,c),1.0)], 0.0, w_perp)
            if 0 < c < W - 1 and m[r, c - 1] and m[r, c + 1]:
                add([(idx(r,c-1),1.0),(idx(r,c),-2.0),(idx(r,c+1),1.0)], 0.0, w_planar)

# constraints 7 height match across crease (edge between two interior faces)
crease_band = binary_dilation(crease, np.ones((3, 3)))
for r, c in zip(*np.where(crease_band)):
    for dr, dc in [(1, 0), (0, 1)]:
        f = g = 0; rf = cf = rg = cg = None
        for k in range(1, 3 + 1):
            rr_, cc_ = r - k*dr, c - k*dc
            if 0 <= rr_ < H and 0 <= cc_ < W and face_aug[rr_, cc_] != 0:
                f, rf, cf = face_aug[rr_, cc_], rr_, cc_; break
        for k in range(1, 3 + 1):
            rr_, cc_ = r + k*dr, c + k*dc
            if 0 <= rr_ < H and 0 <= cc_ < W and face_aug[rr_, cc_] != 0:
                g, rg, cg = face_aug[rr_, cc_], rr_, cc_; break
        if f and g and f != g:
            add([(idx(rf, cf), 1.0), (idx(rg, cg), -1.0)], 0.0, w_connect)

# least squares matrix (sparse) and fitting
A = coo_matrix((vals, (rows, cols)), shape=(eq, N)).tocsr()
b = np.array(b)
Y = lsqr(A, b)[0].reshape(H, W)

# recover Z based on Y
xs_img, ys_img = np.meshgrid(np.arange(W), np.arange(H))
X  = xs_img.astype(float)
Yc = Y
Z  = (cos_t * Y - ys_img.astype(float)) / sin_t

# visualizations of extracted features!
fig, ax = plt.subplots()
plt.imshow(edge_mask, cmap='gray_r')
plt.xticks([]), plt.yticks([])
plt.savefig('outputs/edges.jpeg', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots()
dx = 0.05 * grad_mag_downsampled * np.cos(grad_ori_downsampled) # line drawing mechanics 
dy = 0.05 * grad_mag_downsampled * np.sin(grad_ori_downsampled)
ax.plot([xs - dx/2, xs + dx/2], [ys - dy/2, ys + dy/2], color='red', linewidth=0.6)
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlim([0, gx.shape[1]])
ax.set_ylim([0, gx.shape[0]])
ax.set_ylim(ax.get_ylim()[::-1])  
plt.savefig('outputs/gradmag.jpeg', bbox_inches='tight', dpi=300)
plt.close()

plt.imshow(is_object, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.savefig('outputs/figureground.jpeg', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots()
out = np.ones((*edge_mask.shape, 3))
out[edge_mask & is_vertical]   = [1, 0, 0]
out[edge_mask & is_horizontal] = [0, 1, 0]
plt.imshow(out)
plt.xticks([]), plt.yticks([])
plt.savefig('outputs/edgesclassified.jpeg', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots()
rng = np.random.default_rng(0)
palette = rng.uniform(0.25, 1.0, size=(len(keep) + 1, 3)); palette[0] = 0
face_rgb = palette[face_id]
ax.imshow(face_rgb)
ax.set_xticks([]); ax.set_yticks([])
plt.savefig('outputs/faces.jpeg', bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots()
ax.imshow(is_horizontal_face, cmap='Blues', alpha=0.5)
ax.imshow(is_vertical_face, cmap='Reds', alpha=0.5)
ax.set_xticks([]); ax.set_yticks([])
plt.savefig('outputs/labeledfaces.jpeg', bbox_inches='tight', dpi=300)
plt.close()

# visualizations of the reconstruction!
height_scale = 0.5             
Yc = Yc * height_scale
Z  = (cos_t * Yc - ys_img.astype(float)) / sin_t   

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, M, name in zip(axes, [X, Yc, Z], ['X', 'Y', 'Z']):
    im = ax.imshow(M, cmap='gray')
    ax.set_title(name); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.savefig('outputs/worldcoords.jpeg', bbox_inches='tight', dpi=300)
plt.close()

colors = img_array.reshape(-1, 3) / 255.0 
fig = plt.figure(figsize=(12, 5))
for k, (elev, azim) in enumerate([(90, -90), (30, -60), (10, -30)]):
    ax = fig.add_subplot(1, 3, k+1, projection='3d')

    ax.scatter(X.ravel(), Z.ravel(), Yc.ravel(),
               c=colors, s=2, marker='.', linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1, 1, 0.2))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
plt.tight_layout()
plt.savefig('outputs/render3d.jpeg', bbox_inches='tight', dpi=300)
plt.close()