%% create video

video_filename = 'example.avi';
path = '/work/sommerc/Data/Kinect_Bunny_Handheld/'; % TODO: change path to your data path
v = VideoWriter([path, video_filename]);
v.FrameRate = 20;

open(v)

for k=0:232 % TODO: change
    F = imread([path, 'color_', sprintf('%06d',k), '.png']); % TODO: adapt according to your img filenames
    writeVideo(v, F);
end

close(v)

clear all
