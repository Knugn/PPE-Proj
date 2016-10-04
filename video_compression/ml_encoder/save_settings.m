function save_settings()

	% Constants for accessing the channels
	settings.Y = 1;
	settings.Cb = 2;
	settings.Cr = 3;

    % size of the motion vector search window
    settings.window_size = 16;
    % size of the block to use in the motion vector search
    settings.block_size = 16;
 
    % Set to the correct path to look for intermediate results
    settings.image_path = '../inputs/nature/nature.';

	% Path to the output stream
	settings.stream_path = '../outputs/stream_ml_nature.xml';

    % Total number of frames to encode
	settings.number_of_frames = 5;
	
    % Frequency of I-frames (1 = no P-frames)
	settings.i_frame_frequency = 2;

	% Compression settings.quality. 1 is lowest ~100 highest
	settings.quality = 1; 

	% DEBUGGING:
    
    % Set to 1 to compare to debug output from encoder
    settings.COMPARE_TO_DEBUG = 0;
    
	% Set to 1 to save all intermeidate results to /dump/settings.image_path/
    settings.DUMP_DATA = 1;
    
	% Set to 1 to graphically display intermediate results
    settings.DISP_RESULTS = 0;
    
	% Set to 1 to pause after each frame
    settings.PAUSE_AFTER_FRAME = 0;
    
    save('settings.mat', 'settings');
end
