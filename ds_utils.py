#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2020-2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import sys
sys.path.append('../')
sys.path.append("/usr/local/lib/python3.8/site-packages/")
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import ctypes
import sys
from collections import deque
from .DS_fun.is_aarch_64 import is_aarch64
from .DS_fun.bus_call import bus_call
from .DS_fun.FPS import PERF_DATA
import pyds
import cv2
from threading import Thread
import tracemalloc
import time
import numpy as np
import gc
if not is_aarch64():
    import cupy as cp

class det_data:
    def __init__(self,tracker_id) -> None:
        self.tracker_id = tracker_id

class DS_utils:
    def __init__(self,uri_dict=None):
        #tracemalloc.start()

        MAX_DISPLAY_LEN=64
        self.mm_tracker=0
        self.uri_dict=uri_dict
        self.num=0
        self.inferencer=0
        self.tracker=0
        self.OID=dict()
        self.DATA=deque()
        self.MUXER_OUTPUT_WIDTH=1920
        self.MUXER_OUTPUT_HEIGHT=1080
        self.MUXER_BATCH_TIMEOUT_USEC=4000000

        self.perf_data = None
        self.frames = deque()
        Gst.init(None)
        self.perf_data = PERF_DATA(len(uri_dict))
        number_sources=len(uri_dict)

    #***************************************************************************source_pipeline**************************************************************************************

        print("Creating source Pipeline \n ")
        self.pipeline = Gst.Pipeline()
        if not self.pipeline:
            sys.stderr.write(" Unable to create Pipeline \n")
        self.streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
        if not self.streammux:
            sys.stderr.write(" Unable to create NvStreamMux \n")
        self.streammux.set_property('width', 1920)
        self.streammux.set_property('height', 1080)
        self.streammux.set_property('batch-size', number_sources)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.streammux.set_property("gpu-id",0)
        self.streammux.set_property("attach-sys-ts",False)
        self.streammux_queue=Gst.ElementFactory.make("queue","streammux_queue")
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.streammux_queue)

        num=0
        if uri_dict != None:
            for i in uri_dict:
                print("Creating source_bin ",i," \n ")
                uri_name=uri_dict[i]
                print(uri_name)
                if uri_name.find("rtsp://") == 0 :
                    is_live = True
                self.source_bin=self.create_source_bin(i, uri_name)
                if not self.source_bin:
                    sys.stderr.write("Unable to create source bin \n")
                self.pipeline.add(self.source_bin)
                padname="sink_%u" %num
                sinkpad= self.streammux.get_request_pad(padname) 
                if not sinkpad:
                    sys.stderr.write("Unable to create sink pad bin \n")
                srcpad=self.source_bin.get_static_pad("src")
                if not srcpad:
                    sys.stderr.write("Unable to create src pad bin \n")
                srcpad.link(sinkpad)
                num=num+1
        self.streammux.link(self.streammux_queue)
    







    #***************************************************************************probe_pipeline**************************************************************************************

        self.queue1=Gst.ElementFactory.make("queue", "queue1")
        self.queue2=Gst.ElementFactory.make("queue", "queue2")
        self.queue3=Gst.ElementFactory.make("queue", "queue3")

        self.pipeline.add(self.queue1)
        self.pipeline.add(self.queue2)
        self.pipeline.add(self.queue3)
        self.nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        self.filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
        if not self.filter1:
            sys.stderr.write(" Unable to get the caps filter1 \n")
        self.filter1.set_property("caps", caps1)
        self.sink = Gst.ElementFactory.make("fakesink", "nv3d-sink")
        if not self.sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
        self.pipeline.add(self.filter1)
        self.pipeline.add(self.nvvidconv1)
        self.pipeline.add(self.sink)   
        self.nvvidconv1.link(self.queue1)
        self.queue1.link(self.filter1)
        self.filter1.link(self.queue2)
        self.queue2.link(self.sink)
        self.ref_time=time.time()


    def cb_newpad_rtsp(self,decodebin, decoder_src_pad,data):
        print("In cb_newpad\n")
        source_bin=data
        if True:
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")

    def cb_newpad_file(self,decodebin, decoder_src_pad,data):
        print("In cb_newpad\n")
        caps=decoder_src_pad.get_current_caps()
        gststruct=caps.get_structure(0)
        gstname=gststruct.get_name()
        source_bin=data
        features=caps.get_features(0)

        # Need to check if the pad created by the decodebin is for video and not
        # audio.
        print("gstname=",gstname)
        if(gstname.find("video")!=-1):
            # Link the decodebin pad only if decodebin has picked nvidia
            # decoder plugin nvdec_*. We do this by checking if the pad caps contain
            # NVMM memory features.
            print("features=",features)
            if features.contains("memory:NVMM"):
                # Get the source bin ghost pad
                bin_ghost_pad=source_bin.get_static_pad("src")
                if not bin_ghost_pad.set_target(decoder_src_pad):
                    sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            else:
                sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")

    def decodebin_child_added(self,child_proxy,Object,name,user_data):
        print("Decodebin child added:", name, "\n")
        if(name.find("decodebin") != -1):
            Object.connect("child-added",self.decodebin_child_added,user_data)

    def create_source_bin(self,key,uri):
        print("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name=f"source-bin-{key}"
        print(bin_name)
        nbin=Gst.Bin.new(bin_name)
        if not nbin:
            sys.stderr.write(" Unable to create source bin \n")

        # Source element for reading from the uri.
        # We will use decodebin and let it figure out the container format of the
        # stream and the codec and plug the appropriate demux and decode plugins.

        
        
        #uri_decode_bin.set_property("rtsp-reconnect-interval",0)
        
        # We set the input uri to the source element
        
        if uri.find("rtsp://") == 0:
            uri_decode_bin=Gst.ElementFactory.make("nvurisrcbin", "uri-decode-bin")
            if not uri_decode_bin:
                sys.stderr.write(" Unable to create uri decode bin \n")
            uri_decode_bin.set_property("uri",uri)
            uri_decode_bin.set_property("rtsp-reconnect-interval",5)
            uri_decode_bin.connect("pad-added",self.cb_newpad_rtsp,nbin)
            uri_decode_bin.connect("child-added",self.decodebin_child_added,nbin)
        elif uri.find("file://") == 0:
            uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
            if not uri_decode_bin:
                sys.stderr.write(" Unable to create uri decode bin \n")
            uri_decode_bin.set_property("uri",uri)

            uri_decode_bin.connect("pad-added",self.cb_newpad_file,nbin)
            uri_decode_bin.connect("child-added",self.decodebin_child_added,nbin)

        # We need to create a ghost pad for the source bin which will act as a proxy
        # for the video decoder src pad. The ghost pad will not have a target right
        # now. Once the decode bin creates the video decoder and generates the
        # cb_newpad callback, we will set the ghost pad target to the video decoder
        # src pad.
        Gst.Bin.add(nbin,uri_decode_bin)
        bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
        if not bin_pad:
            sys.stderr.write(" Failed to add ghost pad in source bin \n")
            return None
        return nbin










    # def memory_profiler(self):
    #     x=gc.collect()  
    #     #print("1"*1000)
    #     snapshot = tracemalloc.take_snapshot()
    #     top_stats = snapshot.statistics('lineno')

    #     for stat in top_stats[:10]:
    #         print(stat)
    #     return True
    #***************************************************************************PROBE_FOR_JETSON**************************************************************************************


    def sink_pad_buffer_probe_jetson(self,pad, info, u_data):
        frame_number = 0
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            frame_number = frame_meta.frame_num
            

            frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id).copy()
            frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            #print(n_frame_gpu)
            # Initialize cuda.stream object for stream synchronization

            _time_f = int(frame_meta.ntp_timestamp /1E6)/1000
            #print(_time_f)
            #self.DATA.append((frame,_time_f))
            l_obj=frame_meta.obj_meta_list
            src_=list(self.uri_dict.keys())[frame_meta.pad_index]
            #print(src_)
            #print("#"*100)
            if self.inferencer==1:
                dets=dict()
                while l_obj:
                    try: 
                        # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                        # The casting is done by pyds.NvDsObjectMeta.cast()
                        obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        data = pyds.NvDsObjectMeta.cast(obj_meta)
                        x = int(data.rect_params.left)
                        y = int(data.rect_params.top)
                        w = int(data.rect_params.width)
                        h = int(data.rect_params.height)
                        class_id = data.class_id
                        obj_id = obj_meta.object_id
                        infer_id= data.unique_component_id
                    
                        
                        #print((x,y,w,h,class_id,obj_id),f"unique_component_id: {data.unique_component_id}")
                       # if self.mm_tracker==1:
                        if infer_id ==1:
                            if obj_id not in dets:
                                dets[obj_id]={}
                            dets[obj_id]['p_box']=[x,y,w,h]
                            dets[obj_id]['class_id']=class_id
                            if self.mm_tracker==1:
                                if "s_box" not in dets[obj_id].keys():
                                    dets[obj_id]["s_box"]=[]                                    

                        else:
                            
                            box=[x,y,w,h]
                            for i in dets:
                                x0,y0,w0,h0=dets[i]['p_box']
                                if (x+w//2 >= x0 & x+w//2 <= x0+w0) & (y +h//2 >= y0 & y +h//2 <= y0+h0):
                                    dets[i]["s_box"].append(box)
                                         

                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
                self.DATA.append((src_,frame,_time_f,dets))
            else:
                self.DATA.append((src_,frame,_time_f))   
                  


            #print("Frame Number=", frame_number, "Number of Objects=",num_rects,"Vehicle_count=",obj_counter[PGIE_CLASS_ID_VEHICLE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
            # Get frame rate through this probe
            stream_index = "stream{0}".format(frame_meta.pad_index)
            #global perf_data
            
            self.perf_data.update_fps(stream_index)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    #***************************************************************************PROBE_FOR_SERVER**************************************************************************************

    def sink_pad_buffer_probe_server(self,pad, info, u_data):
        
        frame_number = 0
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        # Retrieve batch metadata from the gst_buffer
        # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
        # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break
            frame_number = frame_meta.frame_num
   
            owner = None
            # Getting Image data using nvbufsurface
            # the input should be address of buffer and batch_id
            # Retrieve dtype, shape of the array, strides, pointer to the GPU buffer, and size of the allocated memory
            data_type, shape, strides, dataptr, size = pyds.get_nvds_buf_surface_gpu(hash(gst_buffer), frame_meta.batch_id)
            # dataptr is of type PyCapsule -> Use ctypes to retrieve the pointer as an int to pass into cupy
            ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
            ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
            # Get pointer to buffer and create UnownedMemory object from the gpu buffer
            c_data_ptr = ctypes.pythonapi.PyCapsule_GetPointer(dataptr, None)
            unownedmem = cp.cuda.UnownedMemory(c_data_ptr, size, owner)
            # Create MemoryPointer object from unownedmem, at index 0
            memptr = cp.cuda.MemoryPointer(unownedmem, 0)
            # Create cupy array to access the image data. This array is in GPU buffer
            frame_ = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
            #frame = cp.asnumpy(frame)
            stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
            # Modify the red channel to add blue tint to image
            with stream:
                pass
                #frame_[:, :, 0] = 0.5 * frame_[:, :, 0] + 0.5
            stream.synchronize()
            frame=np.array(frame_.get(stream=stream), copy=True, order='C')
            #print(frame_.shape)
            #frame=frame_.get(stream=stream)
            #del frame_
            frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            #print(n_frame_gpu)
            # Initialize cuda.stream object for stream synchronization

            _time_f = int(frame_meta.ntp_timestamp /1E6)/1000
            #print(_time_f)
            #self.DATA.append((frame,_time_f))
            l_obj=frame_meta.obj_meta_list
            src_=list(self.uri_dict.keys())[frame_meta.pad_index]
            #print(src_)
            #print("#"*100)
            if self.inferencer==1:
                dets={}
                while l_obj:
                    try: 
                        # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                        # The casting is done by pyds.NvDsObjectMeta.cast()
                        obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        data = pyds.NvDsObjectMeta.cast(obj_meta)
                        #print(data)
                        x = int(data.rect_params.left)
                        y = int(data.rect_params.top)
                        w = int(data.rect_params.width)
                        h = int(data.rect_params.height)
                        class_id = data.class_id
                        obj_id = obj_meta.object_id
                        infer_id= data.unique_component_id
                    
                        
                        #print((x,y,w,h,class_id,obj_id),f"unique_component_id: {data.unique_component_id}")
                       # if self.mm_tracker==1:
                        if infer_id ==1:
                            if obj_id not in dets:
                                dets[obj_id]={}
                            dets[obj_id]['p_box']=[x,y,w,h]
                            dets[obj_id]['class_id']=class_id
                            if self.mm_tracker==1:
                                if "s_box" not in dets[obj_id].keys():
                                    dets[obj_id]["s_box"]=[]                                    

                        else:
                            
                            box=[x,y,w,h]
                            for i in dets:
                                x0,y0,w0,h0=dets[i]['p_box']
                                if (x+w//2 >= x0 & x+w//2 <= x0+w0) & (y +h//2 >= y0 & y +h//2 <= y0+h0):
                                    dets[i]["s_box"].append(box)


                            
                        # else:
                        #     dets.append((x,y,w,h,class_id,obj_id,data.unique_component_id))
                           
                        
                                      

                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
                #print(dets)
                self.DATA.append((src_,frame,_time_f,dets))
                
            else:
                self.DATA.append((src_,frame,_time_f))   
            

      


            #print(len(self.DATA))



            #print("Frame Number=", frame_number, "Number of Objects=",num_rects,"Vehicle_count=",obj_counter[PGIE_CLASS_ID_VEHICLE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
            # Get frame rate through this probe
            stream_index = "stream{0}".format(frame_meta.pad_index)
            #global perf_data
            self.perf_data.update_fps(stream_index)
            del frame
            memptr=None
            unownedmem=None
            try:
                l_frame = l_frame.next
            except StopIteration:
                break












            # if time.time()-self.ref_time >=10:
            #     function_variables = locals()
            #     total_size = 0
            #     variable_sizes = {}
                
            #     for var_name, value in function_variables.items():
            #         var_size = sys.getsizeof(value)
            #         total_size += var_size
            #         variable_sizes[var_name] = var_size
                
            #     for var_name, size in variable_sizes.items():
            #         print(f"{var_name}: {size} bytes")

            #     self.ref_time=time.time()












        return Gst.PadProbeReturn.OK
    
    
    def variable_occupancy(self):
        total_size = sys.getsizeof(self)  # Size of the instance itself
        attribute_sizes = {}

        # Calculate size of each attribute
        for attr, value in self.__dict__.items():
            attr_size = sys.getsizeof(value)
            total_size += attr_size
            attribute_sizes[attr] = attr_size  # Store sizes in a dictionary

        return total_size, attribute_sizes
    def read(self): 
        if len(self.DATA)!=0:
            #print("@@@@@@@@@@@@@@@@",len(self.DATA))
            return 1,self.DATA.popleft()
        else:
            return 0,None   
    #***************************************************************************PIPELINE_RUN**************************************************************************************
    def pipeline_run(self):  
       
        Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "universal_pipeline")

        # create an event loop and feed gstreamer bus mesages to it
        self.loop = GLib.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect ("message", bus_call, self.loop)
        nvvidconv_src_pad=self.nvvidconv1.get_static_pad("src")
        if not nvvidconv_src_pad:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            if is_aarch64():
                nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_jetson, 0)
            else:
                
                nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_server, 0)
            # perf callback function to print fps every 5 sec
            #GLib.timeout_add(10000, self.memory_profiler)
            GLib.timeout_add(5000, self.perf_data.perf_print_callback)
            
        # List the sources
        print("Now playing...")
        print("Starting pipeline \n")
        # start play back and listed to events		
        self.pipeline.set_state(Gst.State.PLAYING)
        try:
            self.loop.run()
        except:
            pass
        # cleanup
        print("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)







#***************************************************************************CAPTURE**************************************************************************************#

class Capture(DS_utils):

    def __init__(self,uri_dict):
        super().__init__(uri_dict)
        self.streammux_queue.link(self.nvvidconv1)
        Thread(target=self.pipeline_run,daemon=True).start()




#***************************************************************************INFERENCER**************************************************************************************#



class Inferencer(DS_utils):

    def __init__(self,uri_dict,pgie_config):
        super().__init__(uri_dict)
        print("Creating Pgie \n ")            
        self.pgie=Gst.ElementFactory.make("nvinfer","infer_")
        self.pgie.set_property('config-file-path',pgie_config)     
        self.pipeline.add(self.pgie)
        self.infer_queue=Gst.ElementFactory.make("queue","infer_queue")
        self.pipeline.add(self.infer_queue)
        self.pgie.link(self.infer_queue)
        self.streammux_queue.link(self.pgie)
        self.infer_queue.link(self.nvvidconv1)
        self.inferencer=1
        Thread(target=self.pipeline_run,daemon=True).start()



#***************************************************************************INFERENCER+TRACKER**************************************************************************************#



class Tracker(DS_utils):

    def __init__(self,uri_dict,pgie_config,tracker_config):
        super().__init__(uri_dict)
        print("Creating Pgie \n ")            
        self.pgie=Gst.ElementFactory.make("nvinfer","infer_")
        self.pgie.set_property('config-file-path',pgie_config)     
        self.pipeline.add(self.pgie)
        self.infer_queue=Gst.ElementFactory.make("queue","infer_queue")
        self.pipeline.add(self.infer_queue)
        print("Creating nvtracker \n ")
        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self.tracker:
            sys.stderr.write(" Unable to create tracker \n")
        
        config = configparser.ConfigParser()
        config.read(tracker_config)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)
    
        self.tracker_queue=Gst.ElementFactory.make("queue", "tracker_queue")
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.tracker_queue)
        self.streammux_queue.link(self.pgie)
        self.pgie.link(self.infer_queue)
        self.infer_queue.link(self.tracker)
        self.tracker.link(self.tracker_queue)
        self.tracker_queue.link(self.nvvidconv1)
        self.inferencer=1

        Thread(target=self.pipeline_run,daemon=True).start()


#***************************************************************************FRAME_LIST_INFERENCER**************************************************************************************#

class MM_Tracker(DS_utils):

    def __init__(self,uri_dict,pgie_config,tracker_config):
        super().__init__(uri_dict)
        self.mm_tracker=1


        print("Creating nvtracker \n ")


        self.tracker = Gst.ElementFactory.make("nvtracker", "tracker")
        if not self.tracker:
            sys.stderr.write(" Unable to create tracker \n")
        
        config = configparser.ConfigParser()
        config.read(tracker_config)
        config.sections()

        for key in config['tracker']:
            if key == 'tracker-width' :
                tracker_width = config.getint('tracker', key)
                self.tracker.set_property('tracker-width', tracker_width)
            if key == 'tracker-height' :
                tracker_height = config.getint('tracker', key)
                self.tracker.set_property('tracker-height', tracker_height)
            if key == 'gpu-id' :
                tracker_gpu_id = config.getint('tracker', key)
                self.tracker.set_property('gpu_id', tracker_gpu_id)
            if key == 'll-lib-file' :
                tracker_ll_lib_file = config.get('tracker', key)
                self.tracker.set_property('ll-lib-file', tracker_ll_lib_file)
            if key == 'll-config-file' :
                tracker_ll_config_file = config.get('tracker', key)
                self.tracker.set_property('ll-config-file', tracker_ll_config_file)
    
        self.tracker_queue=Gst.ElementFactory.make("queue", "tracker_queue")
        self.pipeline.add(self.tracker)
        self.pipeline.add(self.tracker_queue)


        print("Creating Pgie \n ")  

        for i,cfg in enumerate(pgie_config):         
            pgie=Gst.ElementFactory.make("nvinfer","infer_%d"%i)
            pgie.set_property('config-file-path',cfg)   
            infer_queue=Gst.ElementFactory.make("queue","infer_queue_%d"%i)
            self.pipeline.add(pgie)
            self.pipeline.add(infer_queue)
            

            if i==0:
                self.streammux_queue.link(pgie)
                pgie.link(infer_queue)
                infer_queue.link(self.tracker)
                self.tracker.link(self.tracker_queue)
            elif i==1:
                self.tracker_queue.link(pgie)
                pgie.link(infer_queue)
            else:
                print("error")
                break



        #infer_queue.link(self.tracker)
        #self.infer_queue.link(self.tracker)
        
        infer_queue.link(self.nvvidconv1)
        self.inferencer=1

        Thread(target=self.pipeline_run,daemon=True).start()










        