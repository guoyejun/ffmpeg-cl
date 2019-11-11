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
#include <stdio.h>
#include <string.h>
#include "libavutil/avassert.h"
#include "dnn_backend_native_layer_depthwiseconv2dnative.h"

int dnn_load_layer_depthwiseconv2dnative(Layer *layer, AVIOContext *model_file_context, int file_size)
{
    DepthwiseConvParams *conv_params;
    int kernel_size;
    int dnn_size = 0;
    conv_params = av_malloc(sizeof(*conv_params));
    if (!conv_params)
        return 0;
    printf("reaching here!\n");
    conv_params->dilation = (int32_t)avio_rl32(model_file_context);
    conv_params->padding_method = (int32_t)avio_rl32(model_file_context);
    conv_params->channel_multiplier = (int32_t)avio_rl32(model_file_context);
    conv_params->input_channel = (int32_t)avio_rl32(model_file_context); 
    conv_params->kernel_size = (int32_t)avio_rl32(model_file_context);
    dnn_size += 20;

    kernel_size = conv_params->input_channel * conv_params->channel_multiplier *
                      conv_params->kernel_size * conv_params->kernel_size;
    dnn_size += kernel_size * 4;

    if (dnn_size > file_size || conv_params->input_channel <= 0 ||
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
    layer->output_operand_index = (int32_t)avio_rl32(model_file_context);
    dnn_size += 8;
    return dnn_size;
}

int dnn_execute_layer_depthwiseconv2dnative(DnnOperand *operands, const int32_t *input_operand_indexes,
                                            int32_t output_operand_index, const void *parameters)
{
	float *output;
    int32_t input_operand_index = input_operand_indexes[0];
    int in_num = operands[input_operand_index].dims[0];
    int in_height = operands[input_operand_index].dims[1];
    int in_width = operands[input_operand_index].dims[2];
    int in_channel = operands[input_operand_index].dims[3];
    const float *input = operands[input_operand_index].data;
    const DepthwiseConvParams *conv_params = (const DepthwiseConvParams *)parameters;

	int pad_size = (conv_params->padding_method == VALID) ? 
					(conv_params->kernel_size - 1) / 2 * conv_params->dilation : 0;
	int out_num = in_num;
	int out_height = in_height - pad_size*2; 
	int out_width = in_width - pad_size*2;
	int out_channel = in_channel * conv_params->channel_multiplier;
	DnnOperand *output_operand = &operands[output_operand_index];

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
    int src_linesize = in_width * in_channel;

    float *temp_out = av_malloc(sizeof(float)*out_height*out_width*out_channel);
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
				            float kernel_data = conv_params->kernel[kernel_y * filter_linesize * conv_params->channel_multiplier + 
				            			kernel_x * in_channel * conv_params->channel_multiplier + ch*conv_params->channel_multiplier + m];
				            temp_out_data += input_pel *kernel_data;
		                }
		            }
		            temp_out[temp_out_loc++] = temp_out_data;
	            }    
	        }
	   	}
	}

    //Concatenate temp_out to obtain output
	int temp_size_wc = out_width * in_channel;
	int temp_size_hwc = out_height * temp_size_wc;
	int temp_size_nhwc = out_num * temp_size_hwc;

	int output_ind = 0;
	for (int n=0; n<out_num; n++){
		for (int h=0; h<out_height; h++){
			for (int w=0; w<out_width; w++){
				for (int ch=0; ch<in_channel; ch++){
					for (int i=0; i<conv_params->channel_multiplier; i++){
						int input_ind = i*temp_size_nhwc + n*temp_size_hwc + h*temp_size_wc + w*in_channel + ch;
						output[output_ind] = temp_out[input_ind];
						output_ind ++;
					}
				}
			}
		}
	}
	av_freep(&temp_out);
	return 0;
}


