import sys
sys.path.append('../')
sys.path.append("/usr/local/lib/python3.8/site-packages/")
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from ctypes import *
import ctypes

import time
import sys
import math
from collections import deque
from DS_fun.is_aarch_64 import is_aarch64
from DS_fun.bus_call import bus_call
from DS_fun.FPS import PERF_DATA
import datetime
import pyds
import numpy as np
import cupy as cp

import base64
from threading import Thread



class Parallel_inferencer:
    def __init__(self,uri_list):
        MAX_DISPLAY_LEN=64
        self.num=0
        self.inferencer=0
        self.tracker=0
        self.OID=dict()
        self.breaker=30
        self.DATA=deque()
        self.MUXER_OUTPUT_WIDTH=1920
        self.MUXER_OUTPUT_HEIGHT=1080
        self.MUXER_BATCH_TIMEOUT_USEC=4000000
        TILED_OUTPUT_WIDTH=1280
        TILED_OUTPUT_HEIGHT=720
        GST_CAPS_FEATURES_NVMM="memory:NVMM"
        OSD_PROCESS_MODE= 0
        OSD_DISPLAY_TEXT= 1
        self.perf_data = None
        self.frames = deque()
        Gst.init(None)
        self.perf_data = PERF_DATA(len(uri_list))
        number_sources=len(uri_list)

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
        self.streammux_queue=Gst.ElementFactory.make("queue","streammux_queue")
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.streammux_queue)


        for i in range(number_sources):
            print("Creating source_bin ",i," \n ")
            uri_name=uri_list[i]
            print(uri_name)
            if uri_name.find("rtsp://") == 0 :
                is_live = True
            self.source_bin=self.create_source_bin(i, uri_name)
            if not self.source_bin:
                sys.stderr.write("Unable to create source bin \n")
            self.pipeline.add(self.source_bin)
            padname="sink_%u" %i
            sinkpad= self.streammux.get_request_pad(padname) 
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad=self.source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)
        self.streammux.link(self.streammux_queue)
        
    #***************************************************************************probe_pipeline**************************************************************************************




    def cb_newpad(self,decodebin, decoder_src_pad,data):
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

    def create_source_bin(self,index,uri):
        print("Creating source bin")

        # Create a source GstBin to abstract this bin's content from the rest of the
        # pipeline
        bin_name="source-bin-%02d" %index
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
            uri_decode_bin.connect("pad-added",self.cb_newpad,nbin)
        elif uri.find("file://") == 0:
            uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
            if not uri_decode_bin:
                sys.stderr.write(" Unable to create uri decode bin \n")
            uri_decode_bin.set_property("uri",uri)

            uri_decode_bin.connect("pad-added",self.cb_newpad,nbin)
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
            #print(n_frame_gpu)
            # Initialize cuda.stream object for stream synchronization

            _time_f = int(frame_meta.ntp_timestamp /1E6)/1000
            #print(_time_f)
            #self.DATA.append((frame,_time_f))
            l_obj=frame_meta.obj_meta_list
            src_=frame_meta.pad_index
            #print("#"*100)
            if self.inferencer==1:
                while l_obj:
                    try: 
                        # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                        # The casting is done by pyds.NvDsObjectMeta.cast()
                        obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        data = pyds.NvDsObjectMeta.cast(obj_meta)
                        x = data.rect_params.left
                        y = data.rect_params.top
                        w = data.rect_params.width
                        h = data.rect_params.height
                        class_id = data.class_id
                        obj_id = obj_meta.object_id
                        self.DATA.append((src_,frame,_time_f,(x,y,w,h,class_id,obj_id)))
                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
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
            frame = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C')
            #print(n_frame_gpu)
            # Initialize cuda.stream object for stream synchronization

            _time_f = int(frame_meta.ntp_timestamp /1E6)/1000
            #print(_time_f)
            #self.DATA.append((frame,_time_f))
            l_obj=frame_meta.obj_meta_list
            src_=frame_meta.pad_index
            #print("#"*100)
            if self.inferencer==1:
                while l_obj:
                    try: 
                        # Note that l_obj.data needs a cast to pyds.NvDsObjectMeta
                        # The casting is done by pyds.NvDsObjectMeta.cast()
                        obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        data = pyds.NvDsObjectMeta.cast(obj_meta)
                        x = data.rect_params.left
                        y = data.rect_params.top
                        w = data.rect_params.width
                        h = data.rect_params.height
                        class_id = data.class_id
                        obj_id = obj_meta.object_id
                        self.DATA.append((src_,frame,_time_f,(x,y,w,h,class_id,obj_id)))
                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
            else:
                self.DATA.append((src_,frame,_time_f))   
            

      


            #print(len(self.DATA))

            stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
            # Modify the red channel to add blue tint to image
            with stream:
                frame[:, :, 0] = 0.5 * frame[:, :, 0] + 0.5
            stream.synchronize()


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
    

    def create_sink(self,i):
        queue1=Gst.ElementFactory.make("queue", "queue1_{}".format(i))
        queue2=Gst.ElementFactory.make("queue", "queue2_{}".format(i))

        self.pipeline.add(queue1)
        self.pipeline.add(queue2)
        nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1_{}".format(i))
        caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
        filter1 = Gst.ElementFactory.make("capsfilter", "filter1_{}".format(i))
        if not filter1:
            sys.stderr.write(" Unable to get the caps filter1 \n")
        filter1.set_property("caps", caps1)
        sink = Gst.ElementFactory.make("fakesink", "nv3d-sink_{}".format(i))
        if not sink:
            sys.stderr.write(" Unable to create nv3dsink \n")
        self.pipeline.add(filter1)
        self.pipeline.add(nvvidconv1)
        self.pipeline.add(sink)   
        nvvidconv1.link(queue1)
        queue1.link(filter1)
        filter1.link(queue2)
        queue2.link(sink)
        nvvidconv_src_pad=nvvidconv1.get_static_pad("src")
        if not nvvidconv_src_pad:
            sys.stderr.write(" Unable to get src pad \n")
        else:
            if is_aarch64():
                nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_jetson, 0)
            else:
                nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_server, 0)
            # perf callback function to print fps every 5 sec
            GLib.timeout_add(5000, self.perf_data.perf_print_callback)
        
        return nvvidconv1



class Inferencer(Parallel_inferencer):
    def __init__(self,uri_list,config_list,tracker_config):
        super().__init__(uri_list)
        self.inferencer=1  
        self.tee=Gst.ElementFactory.make("tee","tee")
        self.pipeline.add(self.tee)
        self.streammux_queue.link(self.tee)

        for n,config in enumerate(config_list):
            pre_gie_queue=Gst.ElementFactory.make("queue","pre_gie_queue_%u"%n)
            self.pipeline.add(pre_gie_queue)
            tee_src_pad=self.tee.get_request_pad("src_%u"%n)
            pgie=Gst.ElementFactory.make("nvinfer","infer_%u"%n)
            self.pipeline.add(pgie)
            pgie.set_property('config-file-path',config)  
            tee_src_pad.link(pre_gie_queue.get_static_pad("sink"))   
            infer_queue=Gst.ElementFactory.make("queue","infer_queue_%u"%n)
            self.pipeline.add(infer_queue)
            tracker = Gst.ElementFactory.make("nvtracker", "tracker_%u"%n)
            self.pipeline.add(tracker)
            if not tracker:
                sys.stderr.write(" Unable to create tracker \n")
            tracker_queue=Gst.ElementFactory.make("queue","tracker_queue_%u"%n)
            self.pipeline.add(tracker_queue)
            tconfig = configparser.ConfigParser()
            tconfig.read(tracker_config)
            tconfig.sections()

            for key in tconfig['tracker']:
                if key == 'tracker-width' :
                    tracker_width = tconfig.getint('tracker', key)
                    tracker.set_property('tracker-width', tracker_width)
                if key == 'tracker-height' :
                    tracker_height = tconfig.getint('tracker', key)
                    tracker.set_property('tracker-height', tracker_height)
                if key == 'gpu-id' :
                    tracker_gpu_id = tconfig.getint('tracker', key)
                    tracker.set_property('gpu_id', tracker_gpu_id)
                if key == 'll-lib-file' :
                    tracker_ll_lib_file = tconfig.get('tracker', key)
                    tracker.set_property('ll-lib-file', tracker_ll_lib_file)
                if key == 'll-config-file' :
                    tracker_ll_config_file = tconfig.get('tracker', key)
                    tracker.set_property('ll-config-file', tracker_ll_config_file)

            pre_gie_queue.link(pgie)
            pgie.link(infer_queue)
            infer_queue.link(tracker)
            tracker.link(tracker_queue)
            tracker_queue.link(self.create_sink(config[:-4]))
        Thread(target=self.pipeline_run,daemon=True).start()
              



