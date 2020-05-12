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

typedef struct OVModel{
    ie_core_t *core;
    ie_network_t *network;
    ie_executable_network_t *exe_network;
} OVModel;

static DNNReturnType set_input_output_ov(void *model, DNNData *input, const char *input_name, const char **output_names, uint32_t nb_output)
{

    return DNN_SUCCESS;
}

static DNNReturnType get_input_ov(void *model, DNNData *input, const char *input_name)
{
    return DNN_SUCCESS;
}

DNNModel *ff_dnn_load_model_ov(const char *model_filename)
{
    DNNModel *model = NULL;
    OVModel *ov_model = NULL;
    IEStatusCode status;
    ie_config_t config = {NULL, NULL, NULL};

    model = av_malloc(sizeof(DNNModel));
    if (!model){
        return NULL;
    }

    ov_model = av_mallocz(sizeof(OVModel));
    if (!ov_model)
        goto err;

    status = ie_core_create("", &ov_model->core);
    if (status != OK)
        goto err;

    status = ie_core_read_network(ov_model->core, model_filename, NULL, &ov_model->network);
    if (status != OK)
        goto err;

    status = ie_core_load_network(ov_model->core, ov_model->network, "CPU", &config, &ov_model->exe_network);
    if (status != OK)
        goto err;

    model->model = (void *)ov_model;
    model->set_input_output = &set_input_output_ov;
    model->get_input = &get_input_ov;

    return model;

err:
    if (model)
        av_freep(&model);
    if (ov_model) {
        if (ov_model->core)
            ie_core_free(&ov_model->core);
        av_freep(&ov_model);
    }
    return NULL;
}

DNNReturnType ff_dnn_execute_model_ov(const DNNModel *model, DNNData *outputs, uint32_t nb_output)
{
    return DNN_SUCCESS;
}

void ff_dnn_free_model_ov(DNNModel **model)
{

}
