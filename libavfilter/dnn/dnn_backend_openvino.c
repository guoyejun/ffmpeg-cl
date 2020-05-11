/*
 * Copyright (c) 2020
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
 * DNN OpenVINO backend implementation.
 */

#include "dnn_backend_openvino.h"
#include "libavformat/avio.h"
#include "libavutil/avassert.h"

#include <openvino/c_api/ie_c_api.h>

DNNModel *ff_dnn_load_model_ov(const char *model_filename)
{
    DNNModel *model = NULL;
    ie_core_t *core = NULL;
    ie_core_create("", &core);

    return model;
}

DNNReturnType ff_dnn_execute_model_ov(const DNNModel *model, DNNData *outputs, uint32_t nb_output)
{
    return DNN_SUCCESS;
}

void ff_dnn_free_model_ov(DNNModel **model)
{

}
