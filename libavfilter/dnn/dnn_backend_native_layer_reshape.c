/*
 * Copyright (c) 2019 Wenqian Xing
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

/**
 * @file
 * DNN native backend implementation.
 */

#include "dnn_backend_native.h"
#include "libavutil/avassert.h"
#include "dnn_backend_native_layer_reshape.h"
#include <stdio.h>

int dnn_load_layer_reshape(Layer *layer, AVIOContext *model_file_context, int file_size)
{
    DnnLayerReshapeParams *params;
    int dnn_size = 0;
    params = av_malloc(sizeof(*params));
    if (!params)
        return 0;

    params->new_numbers = (int32_t)avio_rl32(model_file_context);
    params->new_height = (int32_t)avio_rl32(model_file_context);
    params->new_weight = (int32_t)avio_rl32(model_file_context);
    params->new_channels = (int32_t)avio_rl32(model_file_context);
    dnn_size += 16;

    layer->params = params;
    layer->input_operand_indexes[0] = (int32_t)avio_rl32(model_file_context);
    layer->output_operand_index = (int32_t)avio_rl32(model_file_context);
    dnn_size += 8;

    return dnn_size;
}

int dnn_execute_layer_reshape(DnnOperand *operands, const int32_t *input_operand_indexes,
                              int32_t output_operand_index, const void *parameters)
{
    const float *src;
    float *dst;
    const DnnLayerReshapeParams *reshape_params = (const DnnLayerReshapeParams *)parameters;
    int32_t input_operand = input_operand_indexes[0];
    int number = operands[input_operand].dims[0];
    int height = operands[input_operand].dims[1];
    int width = operands[input_operand].dims[2];
    int channel = operands[input_operand].dims[3];
    int dims_length = calculate_operand_dims_count(&operands[input_operand]);
    
    DnnOperand *output_operand = &operands[output_operand_index];
    output_operand->dims[0] = reshape_params->new_numbers;
    output_operand->dims[1] = reshape_params->new_height;
    output_operand->dims[2] = reshape_params->new_weight;
    output_operand->dims[3] = reshape_params->new_channels;
    int output_dims_length = calculate_operand_dims_count(output_operand);

    if (output_dims_length < 0 ){
        for (int i = 0; i < 4; ++i){
            if (output_operand->dims[i] == -1){
                output_operand->dims[i] = dims_length/(-output_dims_length);
            }
        }
    }
    output_operand->data_type = operands[input_operand].data_type;
    output_operand->length = calculate_operand_data_length(output_operand);
    output_operand->data = av_realloc(output_operand->data, output_operand->length);
    if (!output_operand->data)
        return DNN_ERROR;
    int dims_count = calculate_operand_dims_count(output_operand);

    src = operands[input_operand].data;
    dst = output_operand->data;
    for (int i = 0; i < dims_count; ++i)
        dst[i] = src[i];

    return 0;
}
