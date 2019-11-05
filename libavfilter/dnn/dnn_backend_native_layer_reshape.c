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

int dnn_load_layer_reshape(Layer *layer, AVIOContext *model_file_context, int file_size)
{
    DnnLayerReshapeParams *params;
    int dnn_size = 0;
    params = av_malloc(sizeof(*params));
    if (!params)
        return 0;

    params->new_numbers = avio_rl32(model_file_context);
    params->new_height = avio_rl32(model_file_context);
    params->new_weight = avio_rl32(model_file_context);
    params->new_channels = avio_rl32(model_file_context);
    dnn_size += 16;

    layer->params = params;
    layer->input_operand_indexes[0] = (int32_t)avio_rl32(model_file_context);
    layer->output_operand_index = (int32_t)avio_rl32(model_file_context);
    dnn_size += 8;

    layer->params = params;
    return dnn_size;
}

int dnn_execute_layer_reshape(DnnOperand *operands, const int32_t *input_operand_indexes,
                              int32_t output_operand_index, const void *parameters)
{
    float *output;
    const float *src;
    const DnnLayerReshapeParams *Reshape_params = (const DnnLayerReshapeParams *)parameters;
    int32_t input_operand_index = input_operand_indexes[0];
    int number = operands[input_operand_index].dims[0];
    int height = operands[input_operand_index].dims[1];
    int width = operands[input_operand_index].dims[2];
    int channel = operands[input_operand_index].dims[3];
    const float *input = operands[input_operand_index].data;

    DnnOperand *output_operand = &operands[output_operand_index];
    output_operand->dims[0] = Reshape_params->new_numbers;
    output_operand->dims[1] = Reshape_params->new_height;
    output_operand->dims[2] = Reshape_params->new_weight;
    output_operand->dims[3] = Reshape_params->new_channels;
    output_operand->data_type = operands[input_operand_index].data_type;
    output_operand->length = calculate_operand_data_length(output_operand);
    output_operand->data = av_realloc(output_operand->data, output_operand->length);
    if (!output_operand->data)
        return -1;
    src = output_operand->data;
    output = src;

    return 0;
}
