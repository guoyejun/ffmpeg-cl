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
#include "libavfilter/dnn/dnn_backend_native_layer_depthwiseconv2dnative.h"

#define EPSON 0.00001
static int test_with_same_multiplier(void)
{
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
    dnn_execute_layer_depthwiseconv2dnative(operands, input_indexes, 1, &params);

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
static int test_with_same_dilate(void)
{
    // the input data and expected data are generated with below python code.
    /*
    x = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    np.random.seed(0)
    test_filter = np.random.rand(3,3,3,1).astype('float32')
    y = tf.nn.depthwise_conv2d(x, test_filter, (1,1,1,1), "SAME", rate=(2,2))

    np.random.seed(0)
    data = np.random.rand(1, 5, 6, 3)

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
    float input[1*5*6*3] = {
        0.5488135039273248, 0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047, 0.6458941130666561, 
        0.4375872112626925, 0.8917730007820798, 0.9636627605010293, 0.3834415188257777, 0.7917250380826646, 0.5288949197529045, 
        0.5680445610939323, 0.925596638292661, 0.07103605819788694, 0.08712929970154071, 0.02021839744032572, 0.832619845547938, 
        0.7781567509498505, 0.8700121482468192, 0.978618342232764, 0.7991585642167236, 0.46147936225293185, 0.7805291762864555, 
        0.11827442586893322, 0.6399210213275238, 0.1433532874090464, 0.9446689170495839, 0.5218483217500717, 0.4146619399905236, 
        0.26455561210462697, 0.7742336894342167, 0.45615033221654855, 0.5684339488686485, 0.018789800436355142, 0.6176354970758771, 
        0.6120957227224214, 0.6169339968747569, 0.9437480785146242, 0.6818202991034834, 0.359507900573786, 0.43703195379934145, 
        0.6976311959272649, 0.06022547162926983, 0.6667667154456677, 0.6706378696181594, 0.2103825610738409, 0.1289262976548533, 
        0.31542835092418386, 0.3637107709426226, 0.5701967704178796, 0.43860151346232035, 0.9883738380592262, 0.10204481074802807, 
        0.2088767560948347, 0.16130951788499626, 0.6531083254653984, 0.2532916025397821, 0.4663107728563063, 0.24442559200160274, 
        0.15896958364551972, 0.11037514116430513, 0.6563295894652734, 0.1381829513486138, 0.1965823616800535, 0.3687251706609641, 
        0.8209932298479351, 0.09710127579306127, 0.8379449074988039, 0.09609840789396307, 0.9764594650133958, 0.4686512016477016, 
        0.9767610881903371, 0.604845519745046, 0.7392635793983017, 0.039187792254320675, 0.2828069625764096, 0.1201965612131689, 
        0.29614019752214493, 0.11872771895424405, 0.317983179393976, 0.41426299451466997, 0.06414749634878436, 0.6924721193700198, 
        0.5666014542065752, 0.2653894909394454, 0.5232480534666997, 0.09394051075844168, 0.5759464955561793, 0.9292961975762141
        };
    float kernel[3*3*3*1] = {
        0.5488135, 0.71518934, 0.60276335, 0.5448832, 0.4236548, 0.6458941, 0.4375872, 0.891773, 0.96366274, 0.3834415, 
        0.79172504, 0.5288949, 0.56804454, 0.92559665, 0.071036056, 0.0871293, 0.020218397, 0.83261985, 0.77815676, 
        0.87001216, 0.9786183, 0.7991586, 0.46147937, 0.7805292, 0.11827443, 0.639921, 0.14335328
    };

    float expected_output[1*5*6*3] = {
        0.92155063, 1.0032489, 1.6773888, 0.96712875, 0.7086746, 0.8458484, 1.5796317, 2.2076485, 1.9721388, 1.5527186, 
        2.1109896, 1.6153831, 1.2854073, 1.7830101, 1.6122878, 1.0688943, 1.284692, 0.5446944, 0.63806057, 0.9632909, 
        0.7927331, 0.75502974, 0.7786846, 0.64434075, 0.77529657, 1.5501871, 2.1791198, 1.2114685, 1.9700375, 1.5507123, 
        0.97543776, 1.3641076, 1.4045585, 0.8694469, 1.0521971, 0.9898242, 1.7146164, 2.025602, 2.5627713, 0.9907372, 
        1.3940942, 1.2583323, 2.5104215, 3.0171037, 3.1221292, 1.5993689, 1.7999945, 2.6499016, 1.6795977, 1.640018, 
        1.739491, 1.1616436, 1.9777961, 2.3350286, 0.6082621, 1.0907882, 1.3630953, 1.0047438, 1.0964684, 1.2281044, 
        0.84920126, 1.8156086, 2.2117798, 1.3860523, 1.1387733, 1.8791709, 0.7363788, 0.9629359, 0.7876866, 0.9357504, 
        1.4406276, 0.8771781, 1.2194415, 0.87731797, 1.5693729, 0.7233298, 0.6029828, 0.99162215, 1.4462005, 1.385218, 
        2.3982425, 1.190069, 1.5225756, 1.3315487, 0.9901483, 0.5368038, 0.97553843, 0.8192497, 1.153074, 0.5758807
    };

    float *output;
    
    params.dilation = 2;
    params.padding_method = SAME;
    params.channel_multiplier = 1;
    params.input_channel = 3; 
    params.kernel_size = 3;
    params.kernel = kernel;

    operands[0].data = input;
    operands[0].dims[0] = 1;
    operands[0].dims[1] = 5;
    operands[0].dims[2] = 6;
    operands[0].dims[3] = 3;
    operands[1].data = NULL;

    input_indexes[0] = 0;
    dnn_execute_layer_depthwiseconv2dnative(operands, input_indexes, 1, &params);

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
    dnn_execute_layer_depthwiseconv2dnative(operands, input_indexes, 1, &params);

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
    if (test_with_same_dilate())
        return 1;
    if (test_with_valid())
        return 1;
    return 0;
}
