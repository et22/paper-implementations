import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.animation as animation

def save_flow_frame(u, v, img, path="./outputs/of_frame.png"):
    plt.figure(figsize=(15,15))
    plt.imshow(img)
    plt.set_cmap("bone")
    plt.xticks([])
    plt.yticks([])

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if u[y, x] > 0 or v[y, x] > 0 or u[y, x] < 0 or v[y, x] < 0:
                plt.annotate('', xy=(x+u[y,x], y+v[y,x]), xytext=(x,y), arrowprops=dict(edgecolor='red', facecolor='red',shrink=0.05, width=1, headwidth=5))

    plt.savefig(path)
    plt.close()

def save_flow_gif(us, vs, video, path="./outputs/of_vid.gif"):
    fig = plt.figure(figsize=(15,15))

    def animate(i):
        plt.cla()
        img = video[i]
        u, v = us[i], vs[i]
        plt.imshow(img)
        plt.set_cmap("bone")
        plt.xticks([])
        plt.yticks([])
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                if u[y, x] > 0 or v[y, x] > 0 or u[y, x] < 0 or v[y, x] < 0:
                    plt.annotate('', xy=(x+u[y,x], y+v[y,x]), xytext=(x,y), arrowprops=dict(edgecolor='red', facecolor='red',shrink=0.05, width=1, headwidth=5))
    ani = animation.FuncAnimation(fig, animate, repeat=True, frames=video.shape[0], interval=1)
    writer = animation.PillowWriter(fps=10,
                                 metadata=dict(artist='Me'),
                                 bitrate=1800)
    ani.save(path, writer=writer)
    
def save_frames(frames, nrows=2, path="./outputs/frames.png"):
    ncols = len(frames) // nrows
    fig, axes = plt.subplots(nrows, len(frames) // nrows, figsize=(6, 6))

    for idx, frame in enumerate(frames):
        if nrows > 1:
            plt.sca(axes[idx // nrows, idx % ncols])
        else:
            plt.sca(axes[idx])
        plt.imshow(frame)
        plt.set_cmap("bone")
        plt.xticks([])
        plt.yticks([])

    plt.savefig(path)
    plt.close()

def save_gif(video, stimulus_size, n_time):
    gif_video = []

    for t in range(n_time):
        gif_frame = np.full((stimulus_size, stimulus_size, 3), 0, np.uint8)
        for i in range(3):
            gif_frame[:, :, i] = np.array(video[t, :, :] * 245, dtype=int)
        gif_video.append(gif_frame)
    imageio.mimsave("./outputs/out.gif", gif_video, fps=10, loop=0)

def make_frame(x1, y1, x2, y2, stimulus_size, square_size):
    frame = np.zeros((stimulus_size, stimulus_size))
    frame[y1 - square_size // 2:y1 + square_size // 2, x1 - square_size // 2:x1 + square_size // 2] = 1
    frame[y2 - square_size // 2:y2 + square_size // 2, x2 - square_size // 2:x2 + square_size // 2] = 1
    return frame

def infer_position(x1, y1, x2, y2, mdir1, mdir2, t):
    y1 = mdir1[0] * t + y1
    y2 = mdir2[0] * t + y2
    x1 = mdir1[1] * t + x1
    x2 = mdir2[1] * t + x2
    return x1, y1, x2, y2

def apply_1dconv(img, k, axis=1):
    out = np.zeros(img.shape)
    r = len(k) // 2

    if axis == 1:
        # reflection pad the image along axis 1
        padded_axis1 = np.pad(img, [(0, 0), (r, r)], mode='reflect') # pads axis 1 with zeros

        # now convolve along axis 1
        for i, w in enumerate(k):
            out += w * padded_axis1[:, i:i+img.shape[1]]
    
    elif axis == 0:
        # reflection pad the image along axis 0  
        padded_axis0 = np.pad(img, [(r,r), (0,0)], mode='reflect')
    
        # now convolve along axis 0
        for i, w in enumerate(k):
            out += w * padded_axis0[i:i+img.shape[0], :]
    return out

def apply_sep_conv(img, k):
    out1 = apply_1dconv(img, k, axis = 1)
    out0 = apply_1dconv(out1, k, axis = 0)
    return out0

def compute_optic_flow(frame1, frame2, savefigs=True, use_reliable=False, reliable_lambda=0.05, reliable_threshdold=0.00005):
    # a) input frames for computing optic flow

    # b) smooth input frames with gaussian with sigma=1, approximated with 5 tap kernel
    fivetap_kernel = np.array([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0

    frame1_smooth = apply_sep_conv(frame1, fivetap_kernel)
    frame2_smooth = apply_sep_conv(frame2, fivetap_kernel)

    if savefigs:
        save_frames([frame1, frame1_smooth, frame2, frame2_smooth], path="outputs/smoothed_frames.png")

    # c) compute the spatial and temporal image gradients 
    # note: I'm not actually sure if spatial gradients should use frame1 or frame2
    # note: does not exactly match but quite close to textbook figure 
    grad_kernel = np.array([1, -8, 0, 8, -1])/12

    lx = apply_1dconv(frame1_smooth, grad_kernel, axis=1)
    ly = apply_1dconv(frame1_smooth, grad_kernel, axis=0) 
    lt = frame2_smooth - frame1_smooth
    
    if savefigs:
        save_frames([lx, ly, lt], nrows=1, path="outputs/grad_frames.png")

    lx2 = lx ** 2
    lylx = ly * lx
    ly2 = ly ** 2
    lxlt = lx*lt
    lylt = ly*lt

    if savefigs:
        save_frames([lx2, lylx, ly2, lxlt, lylt], nrows=1, path="outputs/gradsquare_frames_presmooth.png")

    lx2_smooth = apply_sep_conv(lx2, fivetap_kernel)
    lylx_smooth = apply_sep_conv(lylx, fivetap_kernel)
    ly2_smooth = apply_sep_conv(ly2, fivetap_kernel)
    lxlt_smooth = apply_sep_conv(lxlt, fivetap_kernel)
    lylt_smooth = apply_sep_conv(lylt, fivetap_kernel)

    if savefigs:
        save_frames([lx2_smooth, lylx_smooth, ly2_smooth, lxlt_smooth, lylt_smooth], nrows=1, path="outputs/gradsquare_frames_postsmooth.png")

    # d) construct A and b, compute A inverse, and solve for optic flow vectors at each pixel
    u = np.zeros(frame1.shape)
    v = np.zeros(frame1.shape)
    for y in range(frame1.shape[0]):
        for x in range(frame1.shape[1]):
            A = np.array([[lx2_smooth[y,x], lylx_smooth[y,x]], [lylx_smooth[y,x], ly2_smooth[y,x]]])
            a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]

            if np.abs((a*d - b*c)) > 1e-8:
                Ainv = 1/(a*d - b*c) * np.array([[d, -b], [-c, a]])

                B = -np.array([lxlt_smooth[y,x], lylt_smooth[y,x]])
                
                of = Ainv @ B
                u[y,x], v[y,x] = of[0], of[1]

                if use_reliable:
                    R = (a*d - b*c) - reliable_lambda * (a + d) ** 2
                    if R < reliable_threshdold:
                        u[y,x], v[y,x] = 0, 0

    # e) visualize optic flow vectors overlaid on the frame
    if savefigs:
        save_flow_frame(u, v, frame1)
    return u, v

def generate_translating_squares_video(stimulus_size=33, n_time=8):
    square1_x, square1_y = 22, 10
    square2_x, square2_y = 12, 24
    square_size = 8

    square1_mdir = (0, 1)
    square2_mdir = (1, -1)

    video = np.zeros((n_time, stimulus_size, stimulus_size))
    for t in range(n_time):    
        x1, y1, x2, y2 = infer_position(square1_x, square1_y, square2_x, square2_y, square1_mdir, square2_mdir, t=t)
        video[t, :, :] = make_frame(x1, y1, x2, y2, stimulus_size, square_size)
    return video

if __name__ == "__main__":
    stimulus_size = 35
    n_time = 8

    ########### translating square stimulus ###########
    ### part 1 - stimulus setup and visualization ###
    # create our moving squares video 
    video = generate_translating_squares_video(stimulus_size=stimulus_size, n_time=n_time)
    save_gif(video, stimulus_size, n_time)

    ### part 2 - gradient-based optic flow ###
    # see https://visionbook.mit.edu/optical_flow.html

    # for first two frames
    frame1 = video[0, :, :]
    frame2 = video[1, :, :]
    u, v = compute_optic_flow(frame1, frame2)

    # for all frames
    us, vs = np.zeros((n_time-1, stimulus_size, stimulus_size)), np.zeros((n_time-1, stimulus_size, stimulus_size))
    for t in range(n_time-1):
        us[t], vs[t] = compute_optic_flow(video[t, :, :], video[t+1, :, :], savefigs=False)

    # make gif with us and vs overlaid on video
    save_flow_gif(us, vs, video[:-1, :, :], path="./outputs/trans_squares_vid.gif")

    us, vs = np.zeros((n_time-1, stimulus_size, stimulus_size)), np.zeros((n_time-1, stimulus_size, stimulus_size))
    for t in range(n_time-1):
        us[t], vs[t] = compute_optic_flow(video[t, :, :], video[t+1, :, :], use_reliable=True, savefigs=False)

    # make gif with us and vs overlaid on video
    save_flow_gif(us, vs, video[:-1, :, :], path="./outputs/trans_squares_vid_reliable.gif")