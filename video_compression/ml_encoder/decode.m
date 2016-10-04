function decode()
    global settings;
    load('settings.mat');

    global Y; global Cb; global Cr;
  	global channel_names;
    
	% Constants for accessing the channelsm
    Y = settings.Y;
	Cb = settings.Cb;
	Cr = settings.Cr;
	channel_names = {'Y','Cb','Cr'};
	
	% File to load
	xml_stream = settings.stream_path;
    

    
	% Get the XML stream for the encoded video
	stream = readStream(xml_stream);
	[width, height] = getStreamSize(stream);
    window_size = getStreamWindowSize(stream);
    block_size = getStreamBlockSize(stream);
	quality = getStreamQuality(stream);
	numberOfFrames = getNumberOfFrames(stream);
    
    % Results storage
    decoded_frames_rgb = cell(numberOfFrames);
    decoded_frames_ycrcb = cell(numberOfFrames);


	disp('------------ DECODE -----------')
	fprintf('>> Stream %s: size: %d x %d, quality: %d, number of frames: %d\n', xml_stream, width, height, quality, numberOfFrames)

	% Decode the frames
	for frame_number=0:numberOfFrames-1
		fprintf('\n>>  Decode frame %d\n', frame_number);
		frame_ycrcb = decode_frame(stream, frame_number);
		frame = getFrame(stream, frame_number);
		type = getFrameType(frame);

		h=figure(frame_number+1);
		clf(h);
		set(h, 'name',sprintf('Decoded frame %d', frame_number));
		subplot(2,2,2)
        imagesc(frame_ycrcb(:,:,1));
        colorbar

        t = sprintf('Frame %d (type %s) Y with no motion compensation.', frame_number, type);
		title(t)

		if (strcmp(type, 'P'))
			disp('    P Frame: process motion vectors')

			% Get the motion vectors
			motion_vectors = getFrameMotionVectors(getFrame(stream, frame_number));
			
			% Get the base frame for the motion vectors
            base_frame = getFrameBaseFrame(frame);
            base_frame_ycrcb = decoded_frames_ycrcb{base_frame+1};
            
            subplot(2,2,1)
			imagesc(uint8(decoded_frames_rgb{base_frame+1}));
			t=sprintf('Base frame %d.', base_frame);
			title(t)
            
			% Decode the motion vectors
			motion_delta = do_motion_vectors(motion_vectors, base_frame_ycrcb, window_size, block_size, frame_number);
			
			% Add the results of the motion vectors to the decoded frame
			frame_ycrcb = frame_ycrcb + motion_delta;

			subplot(2,2,3)
			imagesc(uint8(convertYCbCrtoRGB(motion_delta)));
			title('Motion compensation')
		elseif (strcmp(type, 'I'))
			disp('    I Frame')
			% Don't do anything: just use the decoded frame
		else
			error('>>  INVALID FRAME TYPE: %s', type)
		end
		
		% Convert it to RGB
		frame_rgb = convertYCbCrtoRGB(frame_ycrcb);
		frame_rgb = clamp(frame_rgb,0,255);

		subplot(2,2,4)
		imagesc(uint8(frame_rgb));
        decoded_frames_rgb{frame_number+1}=frame_rgb;
        decoded_frames_ycrcb{frame_number+1}=frame_ycrcb;
		title('Final frame')
		
		fprintf('    Finished frame %d\n', frame_number);
	end
	disp('>>  Finished.')
    
	% Display a short animation of the results
    figure
    for x=1:2
        for f=1:numberOfFrames
            imagesc(uint8(decoded_frames_rgb{f}));
            title(sprintf('Decoded Frame %d', f));
            refresh
            pause(0.5)
        end
    end
    
end


function delta = do_motion_vectors(motion_vectors, base_frame_ycrcb, window_size, block_size, frame_number)
    global settings;
    delta = zeros(size(base_frame_ycrcb));
	width = size(base_frame_ycrcb,2);
	height = size(base_frame_ycrcb,1);

	% How far from the edge we can go since we don't special case the edges
    inset = max(window_size, block_size);
	
	% for every block to look at copy the data from i_frame to p_frame
    % this is the upper left coordinate
	current_block = 1;
    for mx=inset+1:block_size:height-(inset+window_size)+1
        for my=inset+1:block_size:width-(inset+window_size)+1
			vector = motion_vectors{current_block};
			current_block = current_block + 1;
			
			% copy the block
			for x=0:block_size-1
				for y=0:block_size-1
					src_x = mx+vector(1)+x;
					src_y = my+vector(2)+y;
					dst_x = mx+x;
					dst_y = my+y;
					delta(dst_x, dst_y, :) =  base_frame_ycrcb(src_x, src_y, :);
				end
			end
		end
    end
    
    if (settings.COMPARE_TO_DEBUG == 1)
        vector_file = sprintf('dump/%sdata/%d-%s.mat', settings.image_path, int32(frame_number), 'motion_vectors');
        load(vector_file);
        original_motion_vectors = data;
        clear data;
        
        tf = isequal(original_motion_vectors, motion_vectors);
        if (tf == 0)
            disp(sprintf('    MOTION VECTORS ARE DIFFERENT!'))
        else
            disp(sprintf('    Motion vectors check OK.'))
        end   
        
    end
           
end



function ycbcr = decode_frame(stream, frame_number)
    global settings;
    global Y; global Cb; global Cr;
    quality = getStreamQuality(stream);
	
	% Storage
	dc = cell(3);
	decoded_blocks = cell(3);
	un_zig_zagged = cell(3);
	blocks_in_place = cell(3);
	decoded = cell(3);
	dequantized = cell(3);
	idct = cell(3);

	% Get the frame data
	frame = getFrame(stream, frame_number);

	if (settings.COMPARE_TO_DEBUG == 1)
 		quant_file = sprintf('dump/%sdata/%d-%s.mat', settings.image_path, int32(frame_number), 'frame_quant');
        load(quant_file);
        i_frame_quant = data;
        
        dct_file = sprintf('dump/%sdata/%d-%s.mat', settings.image_path, int32(frame_number), 'frame_dct');
        load(dct_file);
        i_frame_dct = data;
        
        downsampled_file = sprintf('dump/%sdata/%d-%s.mat', settings.image_path, int32(frame_number), 'frame_downsampled');
        load(downsampled_file);
        i_frame_downsampled = data;
        
        ycbcr_file = sprintf('dump/%sdata/%d-%s.mat', settings.image_path, int32(frame_number), 'frame_ycbcr');
        load(ycbcr_file);
        i_frame_ycbcr = data;
        
        clear data;
	end
	
	% Process each channel
	for channel=Y:Cr
		fprintf('\tProcessing channel: %s\n', getChannelName(channel));

		% Get the size
		[width, height] = getStreamSize(stream);
		
		% Adjust the chroma channels to have half the resolution
		if (channel == Cr || channel == Cb) 
			width = width/2;
			height = height/2;
		end
		
		% Do the processing
		dc_data = getFrameDC(frame, channel);
		block_data = getFrameBlocks(frame, channel);
		dc{channel} = decodeDCcoeff(dc_data, width, height);
		decoded_blocks{channel} = decodeACBlocks(block_data);
		un_zig_zagged{channel} = unZigZagBlocks(decoded_blocks{channel});
		blocks_in_place{channel} = putDecodedBlocksInPlace(un_zig_zagged{channel}, width, height);
		decoded{channel} = blocks_in_place{channel}+dc{channel};
		dequantized{channel} = dequant8x8(decoded{channel}, quality);
		idct{channel} = idct8x8(dequantized{channel});

		if (settings.COMPARE_TO_DEBUG == 1)
			figure(1)
			compare(decoded{channel}, i_frame_quant{channel}, 'Quantized DCT - DC and AC')
			figure(2)
			compare(dequantized{channel}, i_frame_dct{channel}, 'Dequantized DCT - DC and AC')
			figure(3)
			compare(idct{channel}, i_frame_downsampled{channel}, 'Full')
		end
	end

	% Up-sample the chroma channels to have the full resolution image
	upsampled_ycbcr{Y} = idct{Y};
	upsampled_ycbcr{Cr} = upsample(idct{Cr});
	upsampled_ycbcr{Cb} = upsample(idct{Cb});

	ycbcr(:,:,Y) = upsampled_ycbcr{Y};
	ycbcr(:,:,Cb) = upsampled_ycbcr{Cb};
	ycbcr(:,:,Cr) = upsampled_ycbcr{Cr};

	if (settings.COMPARE_TO_DEBUG == 1)
		compare(uint8(ycbcr), uint8(i_frame_ycbcr), 'Final YCbCr')
	end
end


% This function should compare the data from the encoder intermediate steps
% to the decoded values here.
function compare(decoded, original, name)
		subplot(2,2,1)
		imagesc(original)
		colorbar
		title(sprintf('Original %s', name))
		subplot(2,2,2)
		imagesc(decoded)
		colorbar
		title(sprintf('Decoded %s', name))
		subplot(2,2,3)
		imagesc((original-decoded))
		colorbar
		title('ABS diff')
		
		% Check the difference
        DIFF_THRESHOLD = 10;
		max_diff = max(max(abs(original-decoded)));
		if (max_diff > DIFF_THRESHOLD)
			disp(sprintf('\t\t%s max difference: %d', name, max_diff))

            disp(sprintf('...The difference between the frames "%s" is too big, press any key to continue...', name))
			pause
		else
			disp(sprintf('\t\t%s verified', name))
		end
end


function stream = readStream(file_name)
    try
        stream = xmlread(file_name);
    catch
        error('Failed to read XML file %x.', file_name);
    end
end

function numberOfFrames = getNumberOfFrames(stream)
    a = stream.getDocumentElement;
    numberOfFrames = 0;
    for i=0:a.getLength-1
        if strcmp(a.item(i).getNodeName, 'FRAME')
            numberOfFrames = numberOfFrames + 1;
        end
    end
end

function frame = getFrame(stream, frameNumber)
 a = stream.getDocumentElement;
    numberOfFrames = 0;
    for i=0:a.getLength-1
        if strcmp(a.item(i).getNodeName, 'FRAME')    
            if (numberOfFrames == frameNumber)
                frame = a.item(i);
                return
			end
			numberOfFrames = numberOfFrames + 1;
        end
     frame = [];
    end
end

function type = getFrameType(frame)
    type = frame.getAttribute('type').toCharArray();
end

function base_frame = getFrameBaseFrame(frame)
    base_frame = str2double(frame.getAttribute('base'));
end

function num = getFrameNumber(frame)
    num = str2double(frame.getAttribute('number'));
end

function [width, height] = getStreamSize(stream)
    a = stream.getDocumentElement;
    height = str2double(a.getAttribute('height'));
    width = str2double(a.getAttribute('width'));
end

function window_size = getStreamWindowSize(stream)
    a = stream.getDocumentElement;
    window_size = str2double(a.getAttribute('window_size'));
end

function block_size = getStreamBlockSize(stream)
    a = stream.getDocumentElement;
    block_size = str2double(a.getAttribute('block_size'));
end

function quality = getStreamQuality(stream)
	a = stream.getDocumentElement;
	quality = str2double(a.getAttribute('quality'));
end

function dc = getFrameDC(frame, component) 
	search = getChannelName(component);   
    component = getItem(frame, search);
    dc_text = getItem(component, 'DC');
    dc = eval(dc_text.item(0).getData);
end

function blocks = getFrameBlocks(frame, component)
	search = getChannelName(component);   
    component = getItem(frame, search);
    blocksNode = getItem(component, 'BLOCKS');
    blocks = {};
    count = 1;
    for i=0:blocksNode.getLength-1
        if strcmp(blocksNode.item(i).getNodeName, 'B')
            blocks{count} = blocksNode.item(i).item(0).getData; 
            count = count + 1;
        end
    end
end

function name = getChannelName(channel) 
	global channel_names;
	name = channel_names{channel};
end

function motion_vectors = getFrameMotionVectors(frame) 
    vectorNode = getItem(frame, 'MOTION_VECTORS');
    motion_vectors = {};
    count = 1;
    for i=0:vectorNode.getLength-1
        if strcmp(vectorNode.item(i).getNodeName, 'MV')
            motion_vectors{count} = eval(vectorNode.item(i).item(0).getData);
            count = count + 1;
        end
    end
end
    
function subitem = getItem(node, name)
    for i=0:node.getLength-1
        if strcmp(node.item(i).getNodeName, name)
            subitem = node.item(i);
            return
        end
    end
    subitem = [];
end


function dc_coeff_in_place = decodeDCcoeff(frame_dc_data, width, height)
    dc_coeff_in_place = zeros(height, width);

    sum = 0;
    count = 1;
    for x=1:8:height
        for y=1:8:width
            sum = sum + frame_dc_data(count);
            dc_coeff_in_place(x,y) = sum;
            count = count + 1;
        end
    end
end

function block_coeff_in_place = decodeACBlocks(frame_block_data)
    number_of_blocks = size(frame_block_data, 2);
    block_coeff_in_place = cell(number_of_blocks, 1);

    block_size = 8;
    block_length = block_size*block_size;
   
    for i=1:number_of_blocks
       block = frame_block_data{i};
       char_array = (block.toCharArray)';
       split = strsplit(char_array);
       decoded_block = zeros(block_length,1);
       index = 2;
       for b=1:size(split,2)
          value = split(b);
          value = value{1};
          if isempty(strfind(value, 'Z'))
              % Not a repeated zero
              decoded_block(index) = str2num(value);
              index = index + 1;
          else
              % repeated zero
              z_loc = strfind(value, 'Z');
              count_start = z_loc(1);
              count = str2num(value(count_start+1:end));
              for c=1:count
                  decoded_block(index) = 0;
                  index = index + 1;
              end
          end
       end
       if (index~=65)
           error('Did not generate 64 coefficients');
       end
       block_coeff_in_place{i} = decoded_block;
    end
    
end

function blocks = unZigZagBlocks(decoded_blocks)
    % Loads the zigZagIndex variable from a file. This let us change
    % the order without changing the code.
    load('zigZagindex.mat');
	blocks=cell(size(decoded_blocks));
	for i=1:size(decoded_blocks,1)
		zigZagged = decoded_blocks{i};
		unZigZagged = zeros(size(zigZagged));
		for k=1:size(unZigZagged,1)
			unZigZagged(zigZagIndex(k)) = zigZagged(k);
		end
		blocks{i} = unZigZagged;
	end
end

function blocks_in_place = putDecodedBlocksInPlace(decoded_blocks, width, height)
	block_size=8;
	blocks_in_place = zeros(height, width);
	i=1;
	for bx=1:8:height
			for by=1:8:width
			block = decoded_blocks{i};
			k=1;
			for y=by:by+block_size-1
				for x=bx:bx+block_size-1
					blocks_in_place(x,y) = block(k);
					k=k+1;
				end
			end
			i=i+1;
		end
	end
end

function out = dequant8x8(in, quality)
   % De-Quantizes the data 
    % From http://www.mathworks.se/matlabcentral/fileexchange/15335-jpeg-encoder-decoder/content/JPEG%20Encoder%20Decoder/JPEG%20Encoder%20Decoder/jpeg.m
    load('quantMatrix.mat')
    % Matrix assumes 0-255 range
	
	% Adjust for quality
    quantMatrix = ceil(quantMatrix./quality);    
	
    out = zeros(size(in));
    height = size(in,1);
    width = size(in,2);
    for x=1:8:height
        for y=1:8:width
            block = in(x:x+7,y:y+7);
            quantBlock = block.*quantMatrix;
            out(x:x+7,y:y+7) = quantBlock;
        end
	end
end

function out = idct8x8(in)
% 8x8 block dct on each block
    out = zeros(size(in));
    height = size(in,1);
    width = size(in,2);
    for x=1:8:height
        for y=1:8:width
            block = in(x:x+7,y:y+7);
            decodedBlock = idct8x8_block(block);
            out(x:x+7,y:y+7) = decodedBlock;
        end
	end
	% Offset values by 128 to reset them to right range
	out = out + 128;
end


function out = upsample(in)
	[height, width] = size(in);
	out = zeros(height*2, width*2);
	for x=1:height
		for y=1:width
			out(2*x-1,2*y-1) = in(x,y);
			out(2*x,2*y-1) = in(x,y);
			out(2*x-1,2*y) = in(x,y);
			out(2*x,2*y) = in(x,y);
		end
	end
end

