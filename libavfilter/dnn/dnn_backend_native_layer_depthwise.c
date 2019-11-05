/*
 * Copyright (c) 2019 He Yitao
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/avassert.h"
#include "dnn_backend_native_layer_depthwise.h"
typedef enum {VALID, SAME} DNNConvPaddingParam;
int dnn_load_layer_conv2d(Layer *layer, AVIOContext *model_file_context, int file_size)
{
    DepthwiseConvParams *conv_params;
    int kernel_size;
    int dnn_size = 0;
    conv_params = av_malloc(sizeof(*conv_params));
    if (!conv_params)
        return 0;

    conv_params->dilation = (int32_t)avio_rl32(model_file_context);
    conv_params->padding_method = (int32_t)avio_rl32(model_file_context);
    conv_params->channel_multiplier = (int32_t)avio_rl32(model_file_context);
    conv_params->input_num = (int32_t)avio_rl32(model_file_context); //input num is N of input(NHWC)
    conv_params->kernel_size = (int32_t)avio_rl32(model_file_context);
    dnn_size += 20;

    kernel_size = conv_params->input_num * conv_params->channel_multiplier *
                      conv_params->kernel_size * conv_params->kernel_size;
    dnn_size += kernel_size * 4;

    if (dnn_size > file_size || conv_params->input_num <= 0 ||
        conv_params->channel_multiplier <= 0 || conv_params->kernel_size <= 0){
        av_freep(&conv_params);
        return 0;
    }

    conv_params->kernel = av_malloc(kernel_size * sizeof(float));
    if (!conv_params->kernel) {
        av_freep(&conv_params);
        return 0;
    }
    for (int i = 0; i < kernel_size; ++i) {
        conv_params->kernel[i] = av_int2float(avio_rl32(model_file_context));
    }

    layer->params = conv_params;

    layer->input_operand_indexes[0] = (int32_t)avio_rl32(model_file_context);
    // layer->output_operand_index = (int32_t)avio_rl32(model_file_context);
    dnn_size += 4; // change?
    return dnn_size;
}

int dnn_execute_layer_conv2d(DnnOperand *operands, const int32_t *input_operand_indexes,
                             int32_t output_operand_index, const void *parameters)
{
	float *output;
    int32_t input_operand_index = input_operand_indexes[0];
    int in_number = operands[input_operand_index].dims[0];
    int in_height = operands[input_operand_index].dims[1];
    int in_width = operands[input_operand_index].dims[2];
    int in_channel = operands[input_operand_index].dims[3];
    const float *input = operands[input_operand_index].data;
    const ConvolutionalParams *conv_params = (const ConvolutionalParams *)parameters;

	float input[1*3*3*3] = {
		0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777, 0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192, 0.978618342232764, 0.7991585642167236, 0.46147936225293185, 0.7805291762864555, 0.11827442586893322, 0.6399210213275238, 0.1433532874090464
		// 1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3
	};
	// 0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949
	float kernel[3*3*3*1] = {
		0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292, 0.11827443, 0.639921, 0.14335328
		// 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
	};
	//int out_channel = in_channel * conv_params->channel_multiplier;
	// DNNConvPaddingParam pad_mode = VALID; //SAME or VALID pad mode
	

	

	int pad_size = (conv_params->padding_method == VALID) ? 
					(conv_params->kernel_size - 1) / 2 * conv_params->dilation : 0;
	int out_num = in_num;
	int out_height = in_height - pad_size*2; 
	int out_width = in_width - pad_size*2;
	int out_channel = in_channel * conv_params->channel_multiplier;

    output_operand->dims[0] = out_num;
    output_operand->dims[1] = out_height;
    output_operand->dims[2] = out_width;
    output_operand->dims[3] = out_channel;
    output_operand->data_type = operands[input_operand_index].data_type;
    output_operand->length = calculate_operand_data_length(output_operand);
    output_operand->data = av_realloc(output_operand->data, output_operand->length);
    if (!output_operand->data)
        return -1;
    output = output_operand->data;

	int radius = conv_params->kernel_size >> 1;
    int filter_linesize = conv_params->kernel_size * in_channel;
    int filter_size = conv_params->kernel_size * filter_linesize;
    int src_linesize = in_width * in_channel;


	// printf("padsize %d \n", pad_size);
	// printf("out_size: %d %d %d %d\n",in_num, out_height, out_width, out_channel);
	// printf("in_size: %d %d %d %d\n",in_num, in_height, in_width,in_channel);
	// float *output = malloc(sizeof(float)*out_num*out_height*out_width*out_channel); //change


	float temp_out[out_height*out_width*out_channel];
	int temp_out_loc = 0;
	for (int m = 0; m<conv_params->channel_multiplier; ++m){
	    for (int y = pad_size; y < in_height - pad_size; ++y) {
	        for (int x = pad_size; x < in_width - pad_size; ++x) {
	        	for (int ch=0; ch<in_channel; ++ch){
	        		float temp_out_data = 0.f;
	                for (int kernel_y = 0; kernel_y < conv_params->kernel_size; ++kernel_y) {
	                    for (int kernel_x = 0; kernel_x < conv_params->kernel_size; ++kernel_x) {
				        	int y_pos = y + (kernel_y - radius) * conv_params->dilation;
				            int x_pos = x + (kernel_x - radius) * conv_params->dilation;
				            float input_pel = (x_pos < 0 || x_pos >= in_width || y_pos < 0 || y_pos >= in_height) ? 0.0 :
				                        input[y_pos * src_linesize + x_pos * in_channel + ch];
				            float kernel_data = kernel[kernel_y * filter_linesize * conv_params->channel_multiplier + 
				            			kernel_x * in_channel * conv_params->channel_multiplier + ch*conv_params->channel_multiplier + m];
				            temp_out_data += input_pel *kernel_data;
		                }
		            }
		            temp_out[temp_out_loc++] = temp_out_data;
	            }    
	        }
	   	}
	}

//concatenate
	int i1=out_num;
	int i2=out_height;
	int i3=out_width;
	int i4=out_channel;
	int out_size_wc = out_width*out_channel;
	int out_size_hwc = out_height*out_size_wc;
	int out_size_nhwc = out_num*out_size_hwc;

	int output_ind = 0;
	for (int n=0; n<out_num; n++){
		for (int h=0; h<out_height; h++){
			for (int w=0; w<out_width; w++){
				for (int ch=0; ch<out_channel; ch++){
					for (int i=0; i<conv_params->channel_multiplier; i++){
						int input_ind = i*out_size_nhwc + n*out_size_hwc + h*out_size_wc + w*out_channel + ch;
						output[output_ind] = temp_out[input_ind];
						output_ind ++;
					}
				}
			}
		}
	}
	// for(int i=0; i<conv_params->channel_multiplier*out_size_nhwc; i++){
	// 	printf("%f ", output[i]);
	// }	
	return 0;
}


