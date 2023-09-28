#!/usr/bin/env python3

import utils.database as DB 
import os
import signal
import sys

import time
import json

import threading
import websockets
import asyncio

import time

import time
import streamer as streamer
from iyolo import get_yolo, get_perf, CLASSES
import classifier
from utils.utils import draw_border

MAX_REPLAY_FRAMES = 500
#############################

server = None
connected_clients = {}
server_loop = None

current_match = -1
current_replay = -1
new_service = -1 
current_record = None
videoTrack = None

timer = 0

async def ws_handler(websocket, path):
    global enable_gc
    client_id = id(websocket)
    connected_clients[client_id] = websocket

    print(client_id, 'connected')
    send_message(client_id, "device_connected", client_id)

    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            # print(data)
            cmd = data['cmd']

            if cmd == 'play_replay':
                handlePlayReplay(data['params'])
            if cmd == 'play_live':
                handlePlayLive()
            elif cmd == 'new_service':
                handleNewService()
            elif cmd == 'new_match':
                handleNewMatch(data['params'])
            elif cmd == 'finish_match':
                handleFinishMatch()
            elif cmd == 'cancel_service':
                handleCancelService()
            elif cmd == 'set_match':
                handleSetMatch(data['params'])
            elif cmd == 'stream_offer':
                streamOfferHandle(data['params'], client_id)
            elif cmd == 'read_matches':
                handleReadMatches()
            elif cmd == 'read_replays':
                handleReadReplays(data['params'])
            

    finally:
        del connected_clients[client_id]
        print(client_id, 'closed')
        send_message(-1, "device_disconnected", client_id)

def between_callback():
    global server_loop

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_loop = loop
    ws_server = websockets.serve(ws_handler, '0.0.0.0', 8077)

    print("ws started")

    loop.run_until_complete(ws_server)
    loop.run_forever() # this is missing
    loop.close()

async def send_message2(websocket_con, message):
    await websocket_con.send(message)

def send_message(id, cmd, params):
    if server == None: return

    resp = {
        "cmd" : cmd,
        "params" : params
    }
    
    respJson = json.dumps(resp)
    if(id >= 0):
        asyncio.run_coroutine_threadsafe(send_message2(connected_clients[id], respJson), server_loop)
    else:
        for client in connected_clients.values():
            asyncio.run_coroutine_threadsafe(send_message2(client, respJson), server_loop)


##############

async def offering(request, client):
    print("processing offer...")
    
    answer = await streamer.offer(request)
    if server == None: return
    print("ANSWER")
    # print(answer)
    # print(type(answer))
    
    await send_message2(connected_clients[client], json.dumps(answer))
    print("answer sent")

def streamOfferHandle(data, client):
    global server_loop
    asyncio.run_coroutine_threadsafe(offering(data, client), server_loop)
#############

def handlePlayReplay(data):
    replay_id = data["replay_id"]
    record_path = DB.read_recordpath(replay_id)
    videoTrack.play_replay(record_path)
    send_message(-1, "video_on_replay", replay_id)

def handlePlayLive():
    videoTrack.play_live()
    send_message(-1, "video_on_live", True)

def handleNewService():
    global new_service
    global current_record
    new_service = True
    current_record = streamer.Recorder()
    videocap.set_record(current_record)
    classifier.start()

def handleNewMatch(data):
    global current_match
    global timer
    timer = 0
    match_name = data["match_name"]
    match_id = DB.create_new_match(match_name)
    current_match = match_id

    callbackMatchSet()

def handleSetMatch(data):
    global current_match
    global timer
    timer = 0
    match_id = data["match_id"]
    current_match = match_id
    callbackMatchSet()

def handleFinishMatch():
    global current_match
    current_match = -1

def handleCancelService():
    finishService()

def handleReadReplays(data):
    match_id = data["match_id"]
    callbackReadReplays(match_id)

def handleReadMatches():
    callbackReadMatches()

# Callbacks
def callbackNewReplayCreated(replay_id):
    global new_service 
    new_service = False
    send_message(-1, "service_done", replay_id)
    callbackReadReplays(current_match)

def callbackMatchSet():
    #read name and return
    name = DB.read_match_name(current_match)
    send_message(-1, "match_set", name)
    callbackReadReplays(current_match)

def callbackReadReplays(match_id):
    replays = DB.read_replays(match_id)
    processed = []
    for up in replays:
        processed.append({
            "id": up[0],
            "time_service": up[1],
            "recording": up[2].split("/")[1:],
            "foul": up[3]
        })
    send_message(-1, "all_replays", processed)

def callbackReadMatches():
    matches = DB.read_all_matches()
    processed = []
    for up in matches:
        processed.append({
            "id": up[0],
            "name": up[1],
            "date": up[2].strftime("%m/%d/%Y, %H:%M:%S")
        })

    send_message(-1, "all_matches", processed)

def callbackTime():
    send_message(-1, "time_sync", timer)
    

# Functions

def finishService(ret = -1):
    global new_service
    global current_record 
    new_service = False
    current_record.finish()
    replay_id = DB.create_new_replay(current_match, str(timer), ret, current_record.get_file_path())
    current_record = None
    callbackNewReplayCreated(replay_id)

def process_dets(dets):
    finish = -100
    if not classifier.is_standby():
        ret = classifier.runtime(dets)
        if ret >= 0:
            send_message(-1, "det_update", classifier.get_state_str())
        if classifier.state == classifier.STATE_DONE_PADDING:
            return classifier.finish
        elif ret == classifier.STATE_DONE:
            finishService(classifier.finish - 5)
    return finish

#############################

YOLO = None

videocap = None

def detect():
    bgr = videocap.read_in()

    dets, draw = YOLO.infer(bgr)

    if process_dets(dets) > 0:
        draw_border(draw, (255,0,0))

    videocap.store_out(draw)
    if current_record != None:
        if current_record.get_frame_count() >= MAX_REPLAY_FRAMES:
            finishService()

def startInference():
    global YOLO

    global videocap
    global videoTrack

    YOLO = get_yolo("nas")

    print(YOLO)
    
    videocap = streamer.VideoCapture()
    videoTrack = streamer.VideoOpencvTrack(videocap)
    streamer.video = videoTrack

    print('inference is running...')
    return 0

def inferenceLoop():
    global timer
    while(True):
        if current_match > 0:
            timer+=1
        detect()

def shutdown():
    global server
    videocap.release()
    server.join()
    sys.exit()

def close_sig_handler(signal, frame):
    shutdown()

signal.signal(signal.SIGINT, close_sig_handler)

server = threading.Thread(target=between_callback, daemon=True)
server.start()

startInference()
inferenceLoop()