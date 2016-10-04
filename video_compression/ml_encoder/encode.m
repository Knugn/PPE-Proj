function encode()
	global frame_number;
	global IMAGE_FIGURE;
	global PROCESSING_FIGURE;
	global MOTION_FIGURE;
	IMAGE_FIGURE = 1;
	PROCESSING_FIGURE = 2;
	MOTION_FIGURE = 3;
    
    global settings;  
    load('settings.mat');
  
    
    number_of_frames = settings.number_of_frames;
    i_frame_frequency = settings.i_frame_frequency;
  
	fprintf('ENCODE: %d frames with intermediate frames ever %d franes from file %s\n', ...
                number_of_frames, i_frame_frequency, settings.image_path);

	% Load all images
	disp('>>  Loading all images')
	frames = cell(number_of_frames);
	for frame_number=0:number_of_frames-1
		frames{frame_number+1} = load_image(frame_number, settings.image_path);
	end
	disp('    Done loading images.')

	% Create the output file stream
	width = size(frames{1}, 2);
	height = size(frames{1}, 1);
	stream = create_xml_stream(width, height, settings.quality, settings.window_size, settings.block_size);

	% Initialize storage for the intermediate results so MATLAB won't complain
	frame_downsampled = cell(1);
	frame_dct = cell(1);
	frame_quant = cell(1);
	frame_dc_diff = cell(1);
    frame_zigzag = cell(1);
	frame_encode = cell(1);

	% Do the encoding for each frame
	disp('>>  Start encoding')
	for frame_number=0:number_of_frames-1
		fprintf(' >  Encode frame %d\n', frame_number)
		disp_results_clear(frame_number);

		% Get the current frame
		frame_rgb = frames{frame_number+1};
		disp_results(1, 2, uint8(frame_rgb), 'Current Frame RGB');

		% Convert the frame to YCbCr
		disp('    Convert to YCbCr')
		frame_ycbcr = convertRGBtoYCbCr(frame_rgb);
		dump_image(frame_ycbcr, 'frame_ycbcr');
		disp_results(1, 3, uint8(frame_ycbcr), 'Current Frame YCbCr');

		% Low pass the chroma components in preparation for downsampling
		disp('    Lowpass chroma channels')
		frame_ycbcr_lowpass = zeros(size(frame_ycbcr));
		% Don't low pass filter the Y component since we don't downsample it
		frame_ycbcr_lowpass(:,:,settings.Y) = frame_ycbcr(:,:,settings.Y);
        frame_ycbcr_lowpass(:,:,settings.Cb) = lowPass(frame_ycbcr(:,:,settings.Cb));
		frame_ycbcr_lowpass(:,:,settings.Cr) = lowPass(frame_ycbcr(:,:,settings.Cr));
		
		dump_image(frame_ycbcr_lowpass, 'frame_ycbcr_lowpass');

		% Compute motion vectors if needed
		if (mod(frame_number, i_frame_frequency) ~= 0)
			disp('    P Frame: compute motion vectors')
			disp_results(1, 1, uint8(previous_frame_rgb), 'Previous Frame RGB');
            
			 % Compute the motion vectors
			motion_vectors = ...
				motionVectorSearch(previous_frame_ycbcr_lowpass, frame_ycbcr_lowpass);

			% Compute the difference
			frame_ycbcr_lowpass_final = ... 
				computeDelta(previous_frame_ycbcr_lowpass, frame_ycbcr_lowpass, motion_vectors);
		else
			disp('    I Frame: no motion vectors')
			motion_vectors = [];
			frame_ycbcr_lowpass_final = frame_ycbcr_lowpass;
        end
        
        disp_results(1, 4, uint8(frame_ycbcr_lowpass_final), 'YCbCr final');
		
        %dump(motion_vectors, 'motion_vectors');

		% Save this one as the new previous frame
        previous_frame_ycbcr_lowpass = frame_ycbcr_lowpass;

        % Keep the RGB around for debugging
		previous_frame_rgb = frame_rgb;

		% Downsample the chroma components
		disp('    Downsample chroma channels')

        % Don't change the Y
		frame_downsampled{settings.Y} = frame_ycbcr_lowpass_final(:,:,settings.Y);
		frame_downsampled{settings.Cb} = downSample(frame_ycbcr_lowpass_final(:,:,settings.Cb));
		frame_downsampled{settings.Cr} = downSample(frame_ycbcr_lowpass_final(:,:,settings.Cr));
		dump_frame(frame_downsampled, 'frame_downsampled');
		
        disp_results(2, 1, frame_downsampled{settings.Y}, 'Y');
		disp_results(2, 4, frame_downsampled{settings.Cb}, 'Cb Downsampled');
		disp_results(2, 7, frame_downsampled{settings.Cr}, 'Cr Downsampled');

		% Convert to frequency domain
		disp('    DCT')
		frame_dct{settings.Y} = dct8x8(frame_downsampled{settings.Y});
		frame_dct{settings.Cb} = dct8x8(frame_downsampled{settings.Cb});
		frame_dct{settings.Cr} = dct8x8(frame_downsampled{settings.Cr});
		
        dump_frame(frame_dct, 'frame_dct');
		
        disp_results(2, 2, frame_dct{settings.Y}, 'Y DCT');
		disp_results(2, 5, frame_dct{settings.Cb}, 'Cb DCT');
		disp_results(2, 8, frame_dct{settings.Cr}, 'Cr DCT');

		% Quantize the data
		disp('    Quantize data')
		frame_quant{settings.Y} = quant8x8(frame_dct{settings.Y}, settings.quality);
		frame_quant{settings.Cb} = quant8x8(frame_dct{settings.Cb}, settings.quality);
		frame_quant{settings.Cr} = quant8x8(frame_dct{settings.Cr}, settings.quality);
		
        dump_frame(frame_quant, 'frame_quant');
		
        disp_results(2, 3, frame_dct{settings.Y}, 'Y DCT Quantized');
		disp_results(2, 6, frame_dct{settings.Cb}, 'Cb DCT Quantized');
		disp_results(2, 9, frame_dct{settings.Cr}, 'Cr DCT Quantized');

		% Extract the DC components and compute the differences
		disp('    Extract DC componenets and compute differences')
		frame_dc_diff{settings.Y} = dcDiff(frame_quant{settings.Y});
		frame_dc_diff{settings.Cb} = dcDiff(frame_quant{settings.Cb});
		frame_dc_diff{settings.Cr} = dcDiff(frame_quant{settings.Cr});
		dump_frame(frame_dc_diff, 'frame_dc_diff');

		% Zig-zag order for zero-counting
		disp('    Zig-zag ordering')
		frame_zigzag{settings.Y} = zigZagOrder(frame_quant{settings.Y});
		frame_zigzag{settings.Cb} = zigZagOrder(frame_quant{settings.Cb});
		frame_zigzag{settings.Cr} = zigZagOrder(frame_quant{settings.Cr});
		dump_zigzag(frame_zigzag, 'frame_zigzag', width, height);

		% encode coefficients
		disp('    Encode coefficients')
		frame_encode{settings.Y} = encode8x8(frame_zigzag{settings.Y});
		frame_encode{settings.Cb} = encode8x8(frame_zigzag{settings.Cb});
		frame_encode{settings.Cr} = encode8x8(frame_zigzag{settings.Cr});

		% output the 
		disp('    Output results')
		stream_frame(stream, frame_number, motion_vectors, frame_number-1, ... 
                        frame_dc_diff, frame_encode);
		write_stream(settings.stream_path, stream);

		if (settings.PAUSE_AFTER_FRAME==1 && frame_number~=(number_of_frames-1))
			disp('...press any key to continue...')
			pause
		end
	end
	disp('>> Finished.')
end

function disp_results_clear(frame_number)
 	global IMAGE_FIGURE;
	global PROCESSING_FIGURE;
	global MOTION_FIGURE;
    global settings;
    
	if (settings.DISP_RESULTS==0)
		return;
	end
	
	h = figure(IMAGE_FIGURE);

    set(h, 'name', sprintf('RGB/YCbCr Frame %d', frame_number));
	h = figure(PROCESSING_FIGURE);

    set(h, 'name', sprintf('DCT/Quantizing Frame %d', frame_number));
	h = figure(MOTION_FIGURE);

    set(h, 'name', sprintf('Motion Vector Search Frame %d', frame_number));
end

function disp_results(main_figure, sub_figure, data, fig_title)
	global IMAGE_FIGURE;
	global PROCESSING_FIGURE;
	global settings;

	if (settings.DISP_RESULTS==0)
		return;
	end
	
	figure(main_figure);
	if (main_figure == IMAGE_FIGURE)
		subplot(2,2,sub_figure)
	elseif (main_figure == PROCESSING_FIGURE)
		subplot(3,3,sub_figure)
	else
		error('Invalid figure');
	end
	imagesc(data);
	title(fig_title);
end

function dump_zigzag(data, name, width, height)
    data1 = data(1);
    data2 = data(2);
    data3 = data(3);
    
    y = zeros(width*height/64, 64);
    cb = zeros((width/2)*(height/2)/64, 64);
    cr = zeros((width/2)*(height/2)/64, 64);
    
    for i=1:256
        y(i,:) = transpose(data1{1,1}{i,1});
    end
    for i=1:64
        cb(i,:) = transpose(data2{1,1}{i,1});
        cr(i,:) = transpose(data3{1,1}{i,1});
    end
    dump(y, strcat(name,'.Y'));
    dump(cb, strcat(name,'.Cb'));
    dump(cr, strcat(name,'.Cr'));
end

function dump_frame(data, name)
    global settings;
	if (settings.DUMP_DATA == 1)
        dump(data{settings.Y}, strcat(name,'.Y'));
        dump(data{settings.Cb}, strcat(name,'.Cb'));
        dump(data{settings.Cr}, strcat(name,'.Cr'));

	end
end

function dump_image(data, name)
    global settings;
	if (settings.DUMP_DATA == 1)
        dump(data(:,:,settings.Y), strcat(name,'.Y'));
        dump(data(:,:,settings.Cb), strcat(name,'.Cb'));
        dump(data(:,:,settings.Cr), strcat(name,'.Cr'));
	end
end

function dump(data, name) 
	global frame_number;
    global settings;

	if (settings.DUMP_DATA == 1)
		save(sprintf('../dump/ml/%d-%s.mat', int32(frame_number), name), 'data', '-ascii');
	end
end


function showImage(image)
% Scale the image range to fit
	min_val = min(min(min(image)));
	if (min_val < 0)
		image=image-min_val;
	end
	image(find(image<0)) = 0.0;
	max_val = max(max(max(image)));
	image=image./max_val;
	image(find(image>1)) = 1.0;
    imagesc(image)
    axis image
end

function image = load_image(number, path)
    fileName = sprintf('%s%d.tiff', path, number);
    if exist(fileName, 'file')
        image_8bit = imread(fileName);
		% Remove any alpha channels
		image_8bit = image_8bit(:,:,1:3);
        image = double(image_8bit);
	else
		error('File %s does not exit.', fileName)
    end
end


function out = lowPass(in)
    % Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
    % E.g., blur
    % weights for neighboring pixels
    a=0.25;
    b=0.5;
    c=0.25;
    width = size(in, 2);
    height = size(in, 1);
    out = in;
    % In X
    for x=2:height-1
        for y=2:width-1
            out(x,y) = a*in(x-1,y)+b*in(x,y)+c*in(x+1,y);
        end
    end
    % In Y
    for x=2:height-1
        for y=2:width-1
            out(x,y) = a*out(x,y-1)+b*out(x,y)+c*out(x,y+1);
        end
	end
end

function out = downSample(in)
% Downsamples by 2 in X and Y
% Take every other pixel and throw out the rest
    out = in(1:2:end, 1:2:end);
end

function out = dct8x8(in)
	% Offset all values by 128 to make the DCT values be in-range
	in = in-128;
    % 8x8 block dct on each block
    out = zeros(size(in));
    width = size(in, 2);
    height = size(in, 1);
    for x=1:8:height
        for y=1:8:width
            block = in(x:x+7,y:y+7);
            quantBlock = dct8x8_block(block);
            out(x:x+7,y:y+7) = quantBlock;
        end
    end
end


function out = quant8x8(in, quality)
   % Quantizes the data 
    % From http://www.mathworks.se/matlabcentral/fileexchange/15335-jpeg-encoder-decoder/content/JPEG%20Encoder%20Decoder/JPEG%20Encoder%20Decoder/jpeg.m
    load('quantMatrix.mat')
    % Matrix assumes 0-255 range
    % Adjust quantitization by quality
    quantMatrix = ceil(quantMatrix./quality) ;   
	
    out = zeros(size(in));
    height = size(in, 1);
    width = size(in, 2);
    for x=1:8:height
        for y=1:8:width
            block = in(x:x+7,y:y+7);
            quantBlock = round(block./quantMatrix);
            out(x:x+7,y:y+7) = quantBlock;
        end
    end
end

function dcDiffList = dcDiff(dctCoeff)
    width = size(dctCoeff,2);
    height = size(dctCoeff,1);
    number_of_dc = width*height/64;
    % every 8th value in both direction
    dc_values = dctCoeff(1:8:end, 1:8:end);
    % To walk through horizontally then vertically we need to transpose
	dc_values = dc_values';
    dcDiffList = zeros(number_of_dc, 1);
    % Go through and calculate the deltas between each DC coefficient
    dcDiffList(1)=dc_values(1);
    for i=2:number_of_dc
        dcDiffList(i) = dc_values(i)-dc_values(i-1);
    end
end

function ordered = zigZagOrder(in)
    load('zigZagIndex.mat')
    width = size(in, 2);
    height = size(in, 1);
    ordered = cell(width*height/64,1);
    % Go through every block
    blockNumber = 1;
    for x=1:8:height
        for y=1:8:width
            block = in(x:x+7,y:y+7);
            % Put the coefficients in zig-zag order
            zigZagOrdered = zeros(64,1);
            for index=1:64
                zigZagOrdered(index) = block(zigZagIndex(index));
            end
            ordered{blockNumber}=zigZagOrdered;
            blockNumber = blockNumber+1;
        end
    end
end

function encoded = encode8x8(ordered) 
% Takes the zigZaged values and changes them to encoded zero counts
    num_blocks = size(ordered,1);
    encoded = cell(num_blocks,1);
    for i=1:num_blocks
        block_encode = {};
        block = ordered{i};
        num_coeff = size(block,1);
        if (num_coeff ~= 64)
            error('Wrong number of coefficients in block: %d coefficients in block %d: %s\n', num_coeff, i, mat2str(block));
        end
        encoded_index = 1;
        in_zero_run = 0;
        zero_count = 0;
        % Skip DC coefficient
        for c=2:num_coeff
            coeff = block(c);
            if (coeff == 0)
                if (in_zero_run == 0)
                    zero_count = 0;
                    in_zero_run = 1;
                end
                zero_count = zero_count + 1;
            else
                if (in_zero_run == 1)
                    in_zero_run = 0;
                    block_encode{encoded_index} = sprintf('Z%d', zero_count);
                    encoded_index = encoded_index+1;
                end
                block_encode{encoded_index} = num2str(coeff);
                encoded_index = encoded_index+1;
            end
        end
        % If we were in a zero run at the end attach it as well.
        if (in_zero_run == 1)
            if (zero_count > 1)
                block_encode{encoded_index} = sprintf('Z%d', zero_count);
            else
        % Single zeros just as 0
                block_encode{encoded_index} = '0';
            end
        end
        encoded{i} = block_encode;
    end
end





function XML = create_xml_stream(width, height, quality, window_size, block_size)
    XML = com.mathworks.xml.XMLUtils.createDocument('STREAM');
    rootNode = XML.getDocumentElement;
    rootNode.setAttribute('width',num2str(width));
    rootNode.setAttribute('height', num2str(height));
	rootNode.setAttribute('quality', num2str(quality));
    rootNode.setAttribute('window_size', num2str(window_size));
    rootNode.setAttribute('block_size', num2str(block_size));
end
    


function stream_frame(XML, frame_number, motion_vectors, ref_frame_number, dc_diff, image_zero)
    global settings;
    
    rootNode = XML.getDocumentElement;
    frameNode = XML.createElement('FRAME');
    rootNode.appendChild(frameNode);
    frameNode.setAttribute('number', num2str(frame_number));
    if (isempty(motion_vectors))
        frameNode.setAttribute('type', 'I');
	else
        frameNode.setAttribute('type', 'P')
        frameNode.setAttribute('base', num2str(ref_frame_number))
        motionVectorsNode = XML.createElement('MOTION_VECTORS');
        num_vectors = size(motion_vectors, 2);
        for i=1:num_vectors
            mvNode = XML.createElement('MV');
            mvNode.setTextContent(mat2str(motion_vectors{i}));
            motionVectorsNode.appendChild(mvNode);
        end
        frameNode.appendChild(motionVectorsNode);
    end
    
    YNode = XML.createElement('Y');
    CrNode = XML.createElement('Cr');
    CbNode = XML.createElement('Cb');
    frameNode.appendChild(YNode);
    frameNode.appendChild(CrNode);
    frameNode.appendChild(CbNode);
    stream_image(XML, YNode, dc_diff{settings.Y}, image_zero{settings.Y})
    stream_image(XML, CbNode, dc_diff{settings.Cb}, image_zero{settings.Cb})
    stream_image(XML, CrNode, dc_diff{settings.Cr}, image_zero{settings.Cr})

end

function stream_image(XML, parentNode, dc_diff, image_zero)
    dcNode = XML.createElement('DC');
    dcNode.setTextContent(mat2str(dc_diff'));
    parentNode.appendChild(dcNode);
    
    blocksNode = XML.createElement('BLOCKS');
	block_count = 0;
    for i=1:size(image_zero,1)
        bNode = XML.createElement('B');
        bNode.setAttribute('id',num2str(i));
        bNode.setTextContent(strjoin(image_zero{i}));
		block_count = block_count+size(image_zero{i},2);
        blocksNode.appendChild(bNode);
	end
	blocksNode.setAttribute('coeffs',num2str(block_count));
    parentNode.appendChild(blocksNode);
end

function write_stream(stream_path, stream)
    xmlwrite(stream_path, stream)
end

% Helper function to iterate through colors for displaying motion vectors
function c = next_color()
    global colors;
    global color_index;
	c=colors(color_index);
	color_index = color_index+1;
	if (color_index>size(colors))
		color_index=1;
	end
end

% Helper function to iterate through colors for displaying motion vectors
function c = curr_color()
    global colors;
    global color_index;
	c=colors(color_index);
end


function h = rect(location, lineWidth, color) 
    h = rectangle('Position',[location(1)-0.5, location(2)-0.5, location(3), location(4)], 'LineWidth', lineWidth, 'EdgeColor',color);
end

function motion_vectors = motionVectorSearch(source, match)
	global settings;
    global MOTION_FIGURE;
    global colors;
	global color_index;
	colors=['b','g','r','c','m','y'];
	color_index=1;
	
    % Set to 1 to pause eachtime a non-zero motion vector is found
	pause_on_each = 0;
    
    % Weights for how much to look at each componenet in looking for a
    % motion vector match
	Y_weight = 0.5;
	Cr_weight = 0.25;
	Cb_weight = 0.25;
	
    % Window size is how much on each side of the block we search
    window_size = settings.window_size;
    block_size = settings.block_size;
    width = size(match, 2);
    height = size(match, 1);
	
	motion_vectors = {};
	
    % How far from the edge we can go since we don't special case the edges
    inset = max(window_size, block_size);
    
	if (settings.DISP_RESULTS == 1)
		figure(MOTION_FIGURE);
		subplot(2,2,1)
		imagesc(uint8(convertYCbCrtoRGB(source)));
		title('Source (where the data comes from)')
		subplot(2,2,2)
        imagesc(uint8(convertYCbCrtoRGB(match)));
		title('Match (where the data goes)')
	end
	b_rect = [];
	
    % for every block to look at in match
    % this is the upper left coordinate
    for mx=inset+1:block_size:height-(inset+window_size)+1
        for my=inset+1:block_size:width-(inset+window_size)+1
			fprintf('    > Search block at: %g,%g...', mx,my);
			
			block_center_x = block_size/2+mx;
			block_center_y = block_size/2+my;
			
			% Keep track of our best matching vector
			best_match_sad = 1e10;
			best_match_location = [0 0];
			
			% Keep track of SAD results for display
			sad_results = zeros(window_size*2+1, window_size*2+1);
			srx=1;
			sry=1;
			if (settings.DISP_RESULTS == 1)
				figure(MOTION_FIGURE);
				subplot(2,2,2)
				% Location we are looking for
                rect([block_center_x block_center_y  1 1], 2.0,curr_color());
                rect([mx my  block_size block_size], 1.0,curr_color());

				% Location we are searching
				subplot(2,2,1)
				if (ishandle(b_rect))
					delete(b_rect)
                end
                b_rect = rect([mx-window_size my-window_size window_size*2+1+block_size-1 window_size*2+1+block_size-1], 3.0, 'g');

				refresh
			end

            % The search range (upper left of the search)
            for sx=mx-window_size:mx+window_size
                for sy=my-window_size:my+window_size
	
                    current_match_sad = 0;
                    % Do the SAD
                    for x=0:block_size-1
                        for y=0:block_size-1
                            match_x = mx+x;
                            match_y = my+y;
                            search_x = sx+x;
                            search_y = sy+y;
                            diff = abs(match(match_x, match_y, :) - source(search_x, search_y, :));
							diff_total = Y_weight*diff(1) + Cb_weight*diff(2) + Cr_weight*diff(3);
							current_match_sad = current_match_sad + diff_total;
                        end
                    end % SAD
					
                    if (current_match_sad <= best_match_sad)
						best_match_sad = current_match_sad;
						best_match_location = [sx-mx sy-my];
					end
																	
					% For displaying results
					sad_results(sry,srx) = current_match_sad;
					sry=sry+1;
				end
				
				% For displaying results
				sry=1;
				srx=srx+1;
            end % search range
			
			motion_vectors{size(motion_vectors, 2)+1} = best_match_location;
			fprintf('best match: %g at %s\n', best_match_sad, mat2str(best_match_location));
			
			if (settings.DISP_RESULTS == 1)
				figure(MOTION_FIGURE);
				subplot(2,2,1)
                rect([mx+best_match_location(1) my+best_match_location(2)  block_size block_size],  1.0,curr_color());				
				
				figure(MOTION_FIGURE);
				subplot(2,2,3)
				imagesc(sad_results)
				title(sprintf('SAD best value %d at (%d %d)', best_match_sad, best_match_location(1), best_match_location(2)));
				rect([block_size+best_match_location(1)+1 block_size+best_match_location(2)+1  1 1], 1.0, 'y');
                colorbar('east')


				% Show the vector if it is not 0,0
				if (best_match_location(1) ~= 0 || best_match_location(2) ~=0)
					subplot(2,2,2)
					arrow([block_center_x+best_match_location(1) block_center_y+best_match_location(2)], [block_center_x block_center_y], 9, 'EdgeColor', curr_color());
                    if (pause_on_each == 1)
                        pause
                    end
                end
                
                next_color();
			end

        end
    end % match blocks
    
end


function delta = computeDelta(source, match, motion_vectors)
    global settings;
    global MOTION_FIGURE;
  
	delta = match;
	width = size(source, 2);
	height = size(source, 1);
	window_size = settings.window_size;
	block_size = settings.block_size;
	% How far from the edge we can go since we don't special case the edges
    inset = max(window_size, block_size);
	
	% for every block to look at copy the data from i_frame to p_frame
    % this is the upper left coordinate
	current_block = 1;
    for mx=inset+1:block_size:height-(inset+window_size)+1
        for my=inset+1:block_size:width-(inset+window_size)+1
 			vector = motion_vectors{current_block};
			
			% copy the block
			for x=0:block_size-1
				for y=0:block_size-1
					src_x = mx+vector(1)+x;
					src_y = my+vector(2)+y;
					dst_x = mx+x;
					dst_y = my+y;
					delta(dst_x, dst_y, :) = delta(dst_x, dst_y, :) - source(src_x, src_y, :);
				end
			end
			
			if (settings.DISP_RESULTS == 1)
				figure(MOTION_FIGURE);
				subplot(2,2,4)
				imagesc(delta(:,:,1));
                colorbar('east')
				title('Delta')
            end
            
            current_block = current_block + 1;
		end
	end
end

