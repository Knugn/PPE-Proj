function compare_encoders()
    global settings;
    load('settings.mat');

    global Y; global Cb; global Cr;
  	global channel_names;
    
	% Constants for accessing the channelsm
    Y = settings.Y;
	Cb = settings.Cb;
	Cr = settings.Cr;
	channel_names = {'Y','Cb','Cr'};
	
	numberOfFrames = settings.number_of_frames;
    
	disp('------------ COMPARE ENCODERS -----------')

	% Decode the frames
	for frame_number=0:numberOfFrames-1
		fprintf('\n>>  Comparing frame %d\n', frame_number);

        % Comparing YCBCR frames
        compare_c_vs_ml_frame(int32(frame_number), 'frame_ycbcr');
        compare_c_vs_ml_frame(int32(frame_number), 'frame_ycbcr_lowpass');        
        compare_c_vs_ml_frame(int32(frame_number), 'frame_downsampled');
        compare_c_vs_ml_frame(int32(frame_number), 'frame_dct');
        compare_c_vs_ml_frame(int32(frame_number), 'frame_quant');
        compare_c_vs_ml_frame(int32(frame_number), 'frame_dc_diff');
        compare_c_vs_ml_frame(int32(frame_number), 'frame_zigzag');
        disp('>>  Finished.')
    
    end
end

function compare_c_vs_ml_frame(frame_number, frame_name)
        global settings;
        
        y_name = strcat(frame_name, '.Y');
        cb_name = strcat(frame_name, '.Cb');
        cr_name = strcat(frame_name, '.Cr');

        c_file = sprintf('../dump/c/%d-%s.mat', frame_number, y_name);
        ml_file = sprintf('../dump/ml/%d-%s.mat', frame_number, y_name);

        c_data{settings.Y} = load(c_file, '-ascii');
        ml_data{settings.Y} = load(ml_file, '-ascii');

        c_file = sprintf('../dump/c/%d-%s.mat', frame_number, cb_name);
        ml_file = sprintf('../dump/ml/%d-%s.mat', frame_number, cb_name);
      
        c_data{settings.Cb} = load(c_file, '-ascii');   
        ml_data{settings.Cb} = load(ml_file, '-ascii');

        c_file = sprintf('../dump/c/%d-%s.mat', frame_number, cr_name);
        ml_file = sprintf('../dump/ml/%d-%s.mat', frame_number, cr_name);

        c_data{settings.Cr} = load(c_file, '-ascii');     
        ml_data{settings.Cr} = load(ml_file, '-ascii');

        compare_frames(c_data{settings.Y}, ml_data{settings.Y}, y_name);
        compare_frames(c_data{settings.Cb}, ml_data{settings.Cb}, cb_name);
        compare_frames(c_data{settings.Cr}, ml_data{settings.Cr}, cr_name);
end

function compare_frames(f1, f2, name)
        
    result = all(abs(f1 - f2) < 0.01);
    if (result)
        disp(sprintf('\t\tCheck for %s frames OK', name));
    else
        disp(sprintf('\t\tProblem with frames after %s!', name));
    end

end