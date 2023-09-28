#!/usr/bin/env python3

import utils.mathis as mt

STATE_INIT = 0
STATE_WAIT_BALL = 1
STATE_INIT_BALL = 2
STATE_WAIT_BALL_HIT = 3

# 
STATE_UNCLASSIFIED = 4
STATE_BALL_HIT = 5
#
STATE_ANCHOR_FLOAT = 6
STATE_ANCHOR_EXIT = 7

STATE_DONE_PADDING = 8
STATE_DONE = 100

NO_UPDATE = -1
STATE_STANDBY = -2

def get_state_str():
    s = state
    if s == STATE_INIT: return "STATE_INIT"
    elif s == STATE_WAIT_BALL: return "STATE_WAIT_BALL"
    elif s == STATE_INIT_BALL: return "STATE_INIT_BALL"
    elif s == STATE_WAIT_BALL_HIT: return "STATE_WAIT_BALL_HIT"
    elif s == STATE_UNCLASSIFIED: return "STATE_UNCLASSIFIED"
    elif s == STATE_BALL_HIT: return "STATE_BALL_HIT"
    elif s == STATE_ANCHOR_FLOAT: return "STATE_ANCHOR_FLOAT"
    elif s == STATE_ANCHOR_EXIT: return "STATE_ANCHOR_EXIT"
    elif s == STATE_DONE_PADDING: return "STATE_DONE_PADDING"
    elif s == STATE_DONE: return "STATE_DONE"
    else: return " "


LABEL_BALL = 0
LABEL_CIRCLE = 1
LABEL_FLOAT = 2
LABEL_SHOE = 3

BALL_WAIT_CONSISTENCE = 5
BALL_WAIT_TRAJECTORY_LIMIT = 0.7 #deg
ANCHOR_FLOAT_IOU_MIN = 0.3
DONE_PADDING = 50
BALL_H_APPEAR = 120
CIRCLE_EXTRA_Y2 = 0

state = STATE_STANDBY

bbox_circle = None
last_bbox_anchor = None
ball_trajectory = None
last_bbox_ball = None

done_padding_count = 0
finish = STATE_UNCLASSIFIED

def is_standby():
    return state == STATE_STANDBY

def start():
    global bbox_circle
    global last_bbox_anchor
    global ball_trajectory
    global last_bbox_ball
    global state
    global finish
    global done_padding_count

    state = 0
    bbox_circle = None
    last_bbox_anchor = None
    ball_trajectory = None
    last_bbox_ball = None
    finish = STATE_UNCLASSIFIED
    done_padding_count = 0

def stop_runtime():
    set_state(STATE_DONE)

def runtime(dets):
    global last_bbox_anchor
    global last_bbox_ball
    global ball_trajectory
    global bbox_circle
    global finish
    global done_padding_count

    dets_count = len(dets)

    if state == STATE_INIT:

        for det in dets:
            current_class = det[4]
            if current_class == LABEL_CIRCLE:
                bbox_circle = (det[0],det[1],det[2]+CIRCLE_EXTRA_Y2,det[3])
                break
        for det in dets:
            current_class = det[4]
            if current_class == LABEL_SHOE:
                if bbox_circle != None:
                    current_bbox = det[:4]

                    if mt.is_ellipse_inside(current_bbox, bbox_circle):
                        print("done")
                        last_bbox_anchor = current_bbox
                        break
        
        if bbox_circle != None and last_bbox_anchor != None:
            set_state(STATE_WAIT_BALL)
            return STATE_WAIT_BALL

    elif state == STATE_WAIT_BALL:
        for det in dets:
            current_class = det[4]
            if current_class == LABEL_BALL:
                current_bbox = det[:4]

                if last_bbox_ball != None:
                    ball_trajectory = mt.calculate_angle(last_bbox_ball, current_bbox)
                    set_state(STATE_WAIT_BALL_HIT)
                    last_bbox_ball = current_bbox
                    return STATE_WAIT_BALL_HIT
                else:
                    if current_bbox[3] >= 480-BALL_H_APPEAR:
                        last_bbox_ball = current_bbox
                break
        
        if test_anchor_float(dets): return STATE_ANCHOR_FLOAT
        update_last_anchor(dets)
        if test_anchor_exit(): return STATE_ANCHOR_EXIT
    
    elif state == STATE_WAIT_BALL_HIT:
        closest_ball, closest_ball_found = mt.find_closest_bbox_with_class(dets, last_bbox_ball, LABEL_BALL)
        
        if closest_ball_found:
            current_ball_trajectory = mt.calculate_angle(last_bbox_ball, closest_ball)
            print(ball_trajectory)
            print(current_ball_trajectory)
            if not mt.is_angle_difference_below_limit(current_ball_trajectory, ball_trajectory, BALL_WAIT_TRAJECTORY_LIMIT):
                set_state(STATE_BALL_HIT)
                return STATE_BALL_HIT
            last_bbox_ball = closest_ball[:4]
            ball_trajectory = current_ball_trajectory
        
        # ball hit first
        if test_anchor_float(dets): return STATE_ANCHOR_FLOAT
        update_last_anchor(dets)
        if test_anchor_exit(): return STATE_ANCHOR_EXIT

    #finish states:
    if state == STATE_BALL_HIT:
        finish = STATE_BALL_HIT
        set_state(STATE_DONE_PADDING)

    elif state == STATE_ANCHOR_FLOAT:
        finish = STATE_ANCHOR_FLOAT
        set_state(STATE_DONE_PADDING)

    elif state == STATE_ANCHOR_EXIT:
        finish = STATE_ANCHOR_EXIT
        set_state(STATE_DONE_PADDING)

    elif state == STATE_DONE_PADDING:
        if done_padding_count < DONE_PADDING:
            done_padding_count += 1
        else: set_state(STATE_DONE)

    elif state == STATE_DONE:
        set_state(STATE_STANDBY)
        return STATE_DONE
    
    return NO_UPDATE

def update_last_anchor(dets):
    global last_bbox_anchor
    last_bbox_anchor, found = mt.find_closest_bbox_with_class(dets, last_bbox_anchor, LABEL_SHOE)[:4]

def test_anchor_exit():
    if not mt.is_ellipse_inside(last_bbox_anchor, bbox_circle):
        set_state(STATE_ANCHOR_EXIT)
        return True
    return False

def test_anchor_float(dets):
    for det in dets:
        current_class = det[4]
        if current_class == LABEL_FLOAT:
            current_bbox = det[:4]
            if mt.is_iou_above_limit(last_bbox_anchor, current_bbox, ANCHOR_FLOAT_IOU_MIN):
                set_state(STATE_ANCHOR_FLOAT)
                return True
    return False

def set_state(new_state):
    global state
    state = new_state

