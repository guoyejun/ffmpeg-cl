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
#include <math.h>
#include "libavfilter/dnn/dnn_backend_native_layer_depthwiseconv2d.h"

#define EPSON 0.00001
static int test_with_same_multiplier(void)
{
    // test under padding mode is 'SAME' and channel multiplier > 1 
    // the input data and expected data are generated with below python code.
    /*
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    np.random.seed(0)
    test_filter = np.random.rand(3,3,3,2).astype('float32')
    y = tf.nn.depthwise_conv2d(x, test_filter, (1,1,1,1), "SAME")

    np.random.seed(0)
    data = np.random.rand(1, 3, 3, 3)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(y, feed_dict={x: data})
    print("input:")
    print(data.shape)
    print(list(data.flatten()))

    print("test_filter:")
    print(test_filter.shape)
    print(list(test_filter.flatten()))

    print("output:")
    print(output.shape)
    print(list(output.flatten()))
    */
    DepthwiseConvParams params;
    DnnOperand operands[2];
    int32_t input_indexes[1];
    float input[1*3*3*3] = {
        0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 
        0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777, 
        0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 
        0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192, 
        0.978618342232764, 0.7991585642167236, 0.46147936225293185, 0.7805291762864555, 0.11827442586893322, 
        0.6399210213275238, 0.1433532874090464
    };
    float kernel[3*3*3*2] = {
        0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 
        0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293,
         0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292, 
         0.11827443, 0.639921, 0.14335328, 0.9446689, 0.5218483, 0.41466194, 0.2645556, 0.7742337, 
         0.45615032, 0.56843394, 0.0187898, 0.6176355, 0.6120957, 0.616934, 0.94374806, 0.6818203, 
         0.3595079, 0.43703195, 0.6976312, 0.06022547, 0.6667667, 0.67063785, 0.21038257, 0.12892629, 
         0.31542835, 0.36371076, 0.57019675, 0.43860152, 0.9883738, 0.10204481
    };

    float expected_output[1*3*3*6] = {
        0.6557406, 1.0027611, 1.351443, 1.8533658, 0.5081677, 0.7243074, 1.2657465, 1.4674101, 
        2.5432868, 2.6480994, 1.661355, 1.6587613, 0.8842416, 1.1097696, 1.4294459, 1.8256472, 
        1.0016593, 1.0421251, 1.5402462, 2.016458, 2.0982327, 2.3710732, 1.7449583, 1.3255883, 
        2.2479439, 2.6192265, 3.3130572, 3.3317776, 1.7407242, 3.032772, 1.5145282, 1.8300358, 
        2.8856344, 2.0753846, 1.8146404, 1.6871552, 0.7939216, 1.9844172, 1.1639297, 1.4684207, 
        0.94553185, 1.2267542, 1.2398384, 2.1414227, 2.5800865, 2.2830472, 1.1587677, 2.2484725, 
        0.9857367, 1.254923, 1.1207465, 1.4854033, 1.1243075, 1.154919
    };

    float *output;
    
    params.dilation = 1;
    params.padding_method = SAME;
    params.channel_multiplier = 2;
    params.input_channel = 3; 
    params.kernel_size = 3;
    params.kernel = kernel;

    operands[0].data = input;
    operands[0].dims[0] = 1;
    operands[0].dims[1] = 3;
    operands[0].dims[2] = 3;
    operands[0].dims[3] = 3;
    operands[1].data = NULL;

    input_indexes[0] = 0;
    dnn_execute_layer_depthwiseconv2d(operands, input_indexes, 1, &params);

    output = operands[1].data;
    for (int i = 0; i < sizeof(expected_output) / sizeof(float); i++) {
        if (fabs(output[i] - expected_output[i]) > EPSON) {
            printf("at index %d, output: %f, expected_output: %f\n", i, output[i], expected_output[i]);
            av_freep(&output);
            return 1;
        }
    }

    av_freep(&output);
    return 0;
}

static int test_with_valid(void)
{
    // test under padding mode is 'VALID'
    // the input data and expected data are generated with below python code.
    /*
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    np.random.seed(0)
    test_filter = np.random.rand(3,3,3,1).astype('float32')
    y = tf.nn.depthwise_conv2d(x, test_filter, (1,1,1,1), "VALID")

    np.random.seed(0)
    data = np.random.rand(1, 3, 3, 3)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    output = sess.run(y, feed_dict={x: data})
    print("input:")
    print(data.shape)
    print(list(data.flatten()))

    print("test_filter:")
    print(test_filter.shape)
    print(list(test_filter.flatten()))

    print("output:")
    print(output.shape)
    print(list(output.flatten()))
    */
    DepthwiseConvParams params;
    DnnOperand operands[2];
    int32_t input_indexes[1];
    float input[1*3*3*3] = {
        0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 
        0.6458941130666561, 0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777, 
        0.7917250380826646, 0.5288949197529045, 0.5680445610939323, 0.925596638292661, 0.07103605819788694, 
        0.08712929970154071, 0.02021839744032572, 0.832619845547938, 0.7781567509498505, 0.8700121482468192, 
        0.978618342232764, 0.7991585642167236, 0.46147936225293185, 0.7805291762864555, 0.11827442586893322, 
        0.6399210213275238, 0.1433532874090464
    };
    float kernel[3*3*3*1] = {
        0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 
        0.96366274, 0.3834415, 0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 
        0.020218397, 0.83261985, 0.77815676, 0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292, 
        0.11827443, 0.639921, 0.14335328
    };

    float expected_output[1*1*1*3] = {
        2.525041, 4.3495874, 4.2746506
    };

    float *output;
    
    params.dilation = 1;
    params.padding_method = VALID;
    params.channel_multiplier = 1;
    params.input_channel = 3; 
    params.kernel_size = 3;
    params.kernel = kernel;

    operands[0].data = input;
    operands[0].dims[0] = 1;
    operands[0].dims[1] = 3;
    operands[0].dims[2] = 3;
    operands[0].dims[3] = 3;
    operands[1].data = NULL;

    input_indexes[0] = 0;
    dnn_execute_layer_depthwiseconv2d(operands, input_indexes, 1, &params);

    output = operands[1].data;
    for (int i = 0; i < sizeof(expected_output) / sizeof(float); i++) {
        if (fabs(output[i] - expected_output[i]) > EPSON) {
            printf("at index %d, output: %f, expected_output: %f\n", i, output[i], expected_output[i]);
            av_freep(&output);
            return 1;
        }
    }

    av_freep(&output);
    return 0;
}

int main(int argc, char **argv)
{
    if (test_with_same_multiplier())
        return 1;
    if (test_with_valid())
        return 1;
    return 0;
}
