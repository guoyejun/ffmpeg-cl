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

#include <stdio.h>
#include "queue.h"
#include "safe_queue.h"
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/thread.h"

#if HAVE_PTHREAD_CANCEL
#define DNNCond pthread_cond_t
#define dnn_cond_init pthread_cond_init
#define dnn_cond_destroy pthread_cond_destroy
#define dnn_cond_signal pthread_cond_signal
#define dnn_cond_wait pthread_cond_wait
#else
#define DNNCond char
static inline int dnn_cond_init(DNNCond *cond, const void *attr) { return 0; }
static inline int dnn_cond_destroy(DNNCond *cond) { return 0; }
static inline int dnn_cond_signal(DNNCond *cond) { return 0; }
static inline int dnn_cond_wait(DNNCond *cond, AVMutex *mutex)
{
    av_assert0(!"should not reach here");
    return 0;
}
#endif

struct _safe_queue {
    queue *q;
    AVMutex mutex;
    DNNCond cond;
};

safe_queue *safe_queue_create(void)
{
    safe_queue *sq = av_malloc(sizeof(*sq));
    if (!sq)
        return NULL;

    sq->q = queue_create();
    if (!sq->q) {
        av_freep(&sq);
        return NULL;
    }

    ff_mutex_init(&sq->mutex, NULL);
    dnn_cond_init(&sq->cond, NULL);
    return sq;
}

void safe_queue_destroy(safe_queue *sq)
{
    if (!sq)
        return;

    queue_destroy(sq->q);
    ff_mutex_destroy(&sq->mutex);
    dnn_cond_destroy(&sq->cond);
    av_freep(&sq);
}

size_t safe_queue_size(safe_queue *sq)
{
    return sq ? queue_size(sq->q) : 0;
}

int safe_queue_push_front(safe_queue *sq, void *v)
{
    int ret;
    ff_mutex_lock(&sq->mutex);
    ret = queue_push_front(sq->q, v);
    dnn_cond_signal(&sq->cond);
    ff_mutex_unlock(&sq->mutex);
    return ret;
}

int safe_queue_push_back(safe_queue *sq, void *v)
{
    int ret;
    ff_mutex_lock(&sq->mutex);
    ret = queue_push_back(sq->q, v);
    dnn_cond_signal(&sq->cond);
    ff_mutex_unlock(&sq->mutex);
    return ret;
}

void *safe_queue_pop_front(safe_queue *sq)
{
    void *value;
    ff_mutex_lock(&sq->mutex);
    while (queue_size(sq->q) == 0) {
        dnn_cond_wait(&sq->cond, &sq->mutex);
    }
    value = queue_pop_front(sq->q);
    dnn_cond_signal(&sq->cond);
    ff_mutex_unlock(&sq->mutex);
    return value;
}
