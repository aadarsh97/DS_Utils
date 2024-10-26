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
#from .DS_fun.bus_call import bus_call
from .DS_fun.FPS import PERF_DATA
import pyds
import os
import time
import cv2
from random import randint
from threading import Thread
import numpy as np
class list_processor:
    def __init__(self,output_queue,pgie_config,tracker_config,MAX_NUM_SOURCES):
        MAX_DISPLAY_LEN=64
       # self.uri_dict=uri_dict
        self.num=0
        self.inferencer=1
        self.tracker=1
        self.DATA=deque()
        self.MUXER_OUTPUT_WIDTH=1920
        self.MUXER_OUTPUT_HEIGHT=1080
        self.MUXER_BATCH_TIMEOUT_USEC=4000000
        self.output_queue=output_queue
        self.inferenced_data=dict()
        self.perf_data = None
        self.frames = deque()
        self.MAX_NUM_SOURCES=MAX_NUM_SOURCES
        self.src_dict=dict()
        self.running_srcs=dict()
        self.active_srcs=0
        self.eos_list=deque()
        Gst.init(None)
        self.caps_in = Gst.Caps.from_string("video/x-raw,format=RGBA,width=1920,height=1080")


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
        self.streammux.set_property('batch-size', MAX_NUM_SOURCES)
        self.streammux.set_property('batched-push-timeout', 4000000)
        self.streammux.set_property("max-latency",10000000)
        self.streammux.set_property("drop-pipeline-eos",1)
        self.streammux.set_property("gpu-id",0)
        self.streammux_queue=Gst.ElementFactory.make("queue","streammux_queue")
        self.pipeline.add(self.streammux)
        self.pipeline.add(self.streammux_queue)
        self.streammux.link(self.streammux_queue)
        
    #***************************************************************************probe_pipeline**************************************************************************************

        self.queue1=Gst.ElementFactory.make("queue", "queue1")
        self.queue2=Gst.ElementFactory.make("queue", "queue2")
       # self.queue3=Gst.ElementFactory.make("queue", "queue3")

        self.pipeline.add(self.queue1)
        self.pipeline.add(self.queue2)
        #self.pipeline.add(self.queue3)
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
      
        self.nvvidconv1.link(self.queue1)
        self.queue1.link(self.filter1)
        self.filter1.link(self.queue2)
        self.queue2.link(self.sink)
        Thread(target=self.pipeline_run,args=[],daemon=True).start()
    def pipeline_run(self):
        
        self.loop = GLib.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect ("message", self.bus_call, self.loop)
        self.pipeline.set_state(Gst.State.PAUSED)
        print("Starting list_processor..........")
        state=self.pipeline.set_state(Gst.State.PLAYING)
        #print(state)
        
        GLib.timeout_add_seconds(1, self.delete_sources,0)
        GLib.timeout_add_seconds(1, self.add_sources,0)
        GLib.timeout_add_seconds(1, self.get_output)
        nvvidconv_src_pad=self.nvvidconv1.get_static_pad("src")
        if not nvvidconv_src_pad:
            # 
            sys.stderr.write(" Unable to get src pad \n")
        else:
            if is_aarch64():
                nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_jetson, 0)
            else:
                #
                try:
                    
                    nvvidconv_src_pad.add_probe(Gst.PadProbeType.BUFFER, self.sink_pad_buffer_probe_server, 0)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type,exc_obj, fname, exc_tb.tb_lineno,e)
            # perf callback function to print fps every 5 sec
            #GLib.timeout_add(5000, self.perf_data.perf_print_callback)        

        try:
            self.loop.run()
        except:
            pass
        # cleanup
        print("Exiting app\n")
        self.pipeline.set_state(Gst.State.NULL)

###############################################################APPSRC#########################################################################
    
    def __call__(self,frames=list()):
        self.src_dict[frames[0]]=frames[1]
        
        
    
    def videoconvert_bin(self,bin_no):
        bin = Gst.Bin.new("videoconvert_bin_%d"%bin_no)
        videoconvert = Gst.ElementFactory.make("nvvideoconvert", "videoconvert")

        caps = Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA,width=1920,height=1080,framerate=20/1")
        # Create elements
        capsfilter = Gst.ElementFactory.make("capsfilter", "filter")
        capsfilter.set_property('caps', caps)

        if not capsfilter or not videoconvert:
            print("Elements not created ")
            return

        # Add elements to bin
        bin.add(capsfilter)
        bin.add(videoconvert)

        # Link elements within the bin
        videoconvert.link(capsfilter)

        # Create a ghost pad for bin
        sink_pad = Gst.GhostPad.new("sink", videoconvert.get_static_pad("sink"))
        src_pad=Gst.GhostPad.new("src", capsfilter.get_static_pad("src"))
        bin.add_pad(sink_pad)
        bin.add_pad(src_pad)
        return bin





    def push_sticky_events(self,appsrc, caps, stream_id):
        # Create and push caps event
        caps_event = Gst.Event.new_caps(caps)
        appsrc.get_static_pad("src").push_event(caps_event)
        
        # Create and push stream start event
        stream_start_event = Gst.Event.new_stream_start(stream_id)
        appsrc.get_static_pad("src").push_event(stream_start_event)
    def create_appsrc(self, src_id):
        # Create a bin to hold elements
        #bin = Gst.Bin.new(f"appsrc_bin_{bin_no}")
        self.running_srcs[src_id]=[]
        # Define the input caps (format, resolution, etc.)
        caps_in = Gst.Caps.from_string("video/x-raw,format=RGB,width=1920,height=1080,framerate=20/1")

        # Create appsrc element (we'll push buffers to this)
        appsrc = Gst.ElementFactory.make("appsrc", f"app_src_{src_id}")
       # appsrc.set_property("handle-segment-change",True)
        if not appsrc:
            print(f"Error: appsrc not created for bin {src_id}")
            return None
        appsrc.set_property('caps', caps_in)
        #appsrc.set_property('stream_type',0)
        self.running_srcs[src_id].append(appsrc)
        
        # Create videoconvert element (for format conversion)
        videoconvert=Gst.ElementFactory.make("nvvideoconvert",f"videoconvert_{src_id}")
        #videoconvert = self.videoconvert_bin(src_id)
        if not videoconvert:
            print(f"Error: videoconvert not created for bin {src_id}")
            return None

        self.running_srcs[src_id].append(videoconvert)
        self.pipeline.add(self.running_srcs[src_id][0])
        self.pipeline.add(self.running_srcs[src_id][1])
        
        self.running_srcs[src_id][0].link(self.running_srcs[src_id][1])
        try:
            mux_sink= self.streammux.get_request_pad(f"sink_{src_id}")
            if not mux_sink:
                print("can't get request pad for {}".format(src_id))
            self.running_srcs[src_id][1].get_static_pad("src").link(mux_sink)
        except Exception as e:
            print("error in create_appsrc",e,"*"*1000)


       
       

      
        

        # Return the constructed bin
        return self.running_srcs[src_id][0]

    def push_frames(self, src_id):
        try:
            appsrc,videoconvert=self.running_srcs[src_id]
            
            #src_bin.get_static_pad("src").link(self.streammux.get_request_pad("sink_%d"%src_id))
                #Set state of source bin to playing
            state_return = appsrc.set_state(Gst.State.PLAYING)
            state_videoconvert=videoconvert.set_state(Gst.State.PLAYING)
            if state_return == Gst.StateChangeReturn.SUCCESS and state_videoconvert == Gst.StateChangeReturn.SUCCESS:
                #print("STATE CHANGE SUCCESS\n")
                #Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "list_processor")
                #buffer_list = Gst.BufferList.new()
                #self.push_sticky_events(appsrc, self.caps_in, str(src_id))
                #pipeline_state=self.pipeline.set_state(Gst.State.PLAYING)
                # while pipeline_state == Gst.StateChangeReturn.ASYNC:
                #     print(pipeline_state,444444444444444444444444444444444444444444444444444444444444444)
                #     time.sleep(0.1)
                if src_id not in self.inferenced_data.keys():         
                    self.inferenced_data[src_id]=dict()
                    self.inferenced_data[src_id]["data"]=list()
                    self.inferenced_data[src_id]["num"]=len(self.src_dict[src_id])
                for _ in range(len(self.src_dict[src_id])):
                    frame = self.src_dict[src_id].popleft()
                    #print(frame)
                    buffer = self.ndarray_to_gst_buffer(frame)
                    appsrc.emit("push-buffer", buffer)
                    #Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "list_processor_{}".format(src_id))
                    #appsrc.emit("push-buffer",Gst.Buffer.new_wrapped(frame.tobytes()))
                self.eos_list.append(src_id)
                #appsrc.emit("end-of-stream")
               # # eos_event = Gst.Event.new_eos()
                # appsrc.get_static_pad("src").push_event(eos_event)
                ##Gst.debug_bin_to_dot_file(self.pipeline, Gst.DebugGraphDetails.ALL, "list_processor_1")
                #source_id += 1


            elif state_return == Gst.StateChangeReturn.FAILURE:
                print("STATE CHANGE FAILURE\n")
            
            # elif state_return == Gst.StateChangeReturn.ASYNC:
            #     state_return = self.running_srcs[source_id].get_state(Gst.CLOCK_TIME_NONE)
                #source_id += 1

            elif state_return == Gst.StateChangeReturn.NO_PREROLL:
                print("STATE CHANGE NO PREROLL\n")

            # self.pipeline.set_state(Gst.State.PLAYING)
            # for _ in range(len(frame_list)):
            #     frame = frame_list.popleft()
            #     buffer = self.ndarray_to_gst_buffer(frame)
            #     self.appsource.emit("push-buffer", buffer)
            # self.appsource.emit("end-of-stream")
            # bus = self.pipeline.get_bus()
            # while True:
            #     msg = bus.timed_pop_filtered(100 * Gst.MSECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
            #     if msg:
            #         t = msg.type
            #         if t == Gst.MessageType.EOS:
            #             break
            #         elif t == Gst.MessageType.ERROR:
            #             err, debug = msg.parse_error()
            #             sys.stderr.write(f"Error received from element {msg.src.get_name()}: {err}\n")
            #             sys.stderr.write(f"Debugging information: {debug}\n")
            #             break

            # self.pipeline.set_state(Gst.State.READY)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno, e)
    def ndarray_to_gst_buffer(self, array: np.ndarray) -> Gst.Buffer:
        """Converts numpy array to Gst.Buffer"""
        return Gst.Buffer.new_wrapped(array.tobytes())
#######################################################################################################################
    def stop_release_source(self,source_id):
 
        
        if self.active_srcs >=2:
                
            #Attempt to change status of source to be released 
            state_return_appsrc = self.running_srcs[source_id][0].set_state(Gst.State.NULL)
            if state_return_appsrc== Gst.StateChangeReturn.SUCCESS or state_return_appsrc==Gst.StateChangeReturn.ASYNC:
                state_return = self.running_srcs[source_id][1].set_state(Gst.State.NULL)
                if state_return == Gst.StateChangeReturn.SUCCESS:
                    #print("STATE CHANGE SUCCESS\n")
                    pad_name = "sink_%u" % source_id
                    
                    #Retrieve sink pad to be released
                    sinkpad = self.streammux.get_static_pad(pad_name)
                    #Send flush stop event to the sink pad, then release from the self.streammux
                    sinkpad.send_event(Gst.Event.new_flush_stop(False))
                    try:
                        self.streammux.release_request_pad(sinkpad)
                    except Exception as e:
                        print("unable to free sink pad from streammux for {}".format(source_id))
                    self.running_srcs[source_id][0].unlink(self.running_srcs[source_id][1])
                    #print("STATE CHANGE SUCCESS\n")
                    
                    #Remove the source bin from the self.pipeline
                    self.pipeline.remove(self.running_srcs[source_id][0])
                    self.pipeline.remove(self.running_srcs[source_id][1])
                    self.active_srcs-=1
                    
                    del self.src_dict[source_id]
                    del self.running_srcs[source_id]
                    
                    print(f"deleted source successfully : {source_id}")
            
                
                
                elif state_return == Gst.StateChangeReturn.ASYNC:
                    state_return = self.running_srcs[source_id].get_state(Gst.CLOCK_TIME_NONE)
                    pad_name = "sink_%u" % source_id
                    sinkpad = self.streammux.get_static_pad(pad_name)
                    sinkpad.send_event(Gst.Event.new_flush_stop(False))
                    self.streammux.release_request_pad(sinkpad)
                    self.running_srcs[source_id][0].unlink(self.running_srcs[source_id][1])                
                    print("STATE CHANGE ASYNC\n")
                    self.pipeline.remove(self.running_srcs[source_id][0])
                    self.pipeline.remove(self.running_srcs[source_id][1])
                    
                
                    del self.running_srcs[source_id]
                    del self.src_dict[source_id]
        else:
            self.eos_list.append(source_id)
    def delete_sources(self,data):
        #print(self.eos_list)


        #First delete sources that have reached end of stream
        keys=self.eos_list
        if len(keys)!=0:
            for _ in range(len(keys)):
                source_id=keys.popleft()
                if (len(self.src_dict[source_id]) ==0):
                    self.stop_release_source(source_id)


        return True


    def add_sources(self,data):
          

        time.sleep(0.001)
        for id in self.src_dict.copy():
            if  self.active_srcs >= self.MAX_NUM_SOURCES:
                break

            if id not in self.running_srcs.keys():
                source_id = id
                

                #Randomly select an un-enabled source to add
        

                #print("Calling Start %d " % source_id)

                #Create a uridecode bin with the chosen source id
                src = self.create_appsrc(source_id)
                
                if (not src):
                    sys.stderr.write("Failed to create source bin. Exiting.")
                    exit(1)
          
                self.active_srcs+=1

                Thread(target=self.push_frames,args=[source_id],daemon=True).start()

        return True
            
            

    def bus_call(self,bus, message, loop):
        t = message.type
        
        if t== Gst.MessageType.STREAM_STATUS:
            pass
            #print("STREAM STATUS",message.parse_stream_status(),"\n")
        if t== Gst.MessageType.STATE_CHANGED:
            pass
            #print("STATE CHANGED",message.parse_state_changed(),"\n")
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            
            if "No Sources found at the input of muxer" not in str(err):
                sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    #print("Got EOS from stream %d" % stream_id)
                    #self.eos_list.append(stream_id)'
                    pass
                                      
        else:
            pass
           # print(t)
                    
        return True
#######################################################################################################################

   

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
            #src_=list(self.uri_dict.keys())[frame_meta.pad_index]
            #print(src_)
            #print("#"*100)
            stream_id = frame_meta.pad_index

            if self.inferencer==1:
                dets=list()
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
                        dets.append((x,y,w,h,class_id,obj_id))
                        #print(frame_meta.pad_index,x,y,w,h,class_id,obj_id)
                        #self.DATA.append((src_,frame,_time_f,(x,y,w,h,class_id,obj_id)))
                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
                self.inferenced_data[stream_id]["data"].append((frame,_time_f,dets))
                self.inferenced_data[stream_id]["data"]-=1
            else:
                pass
  
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    #***************************************************************************PROBE_FOR_SERVER**************************************************************************************

    def sink_pad_buffer_probe_server(self,pad, info, u_data):
        #print(555555555555555555555555555555555555555555555555555555555555555555)
        import cupy as cp
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
            #print(frame_number,444444444444444444444444444444444444444444444444444444)
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
            frame = cp.ndarray(shape=shape, dtype=data_type, memptr=memptr, strides=strides, order='C').copy()
            frame = cp.asnumpy(frame)
            frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
            stream_id= frame_meta.pad_index
            #cv2.imwrite(f"image/{stream_index}_{randint(1,200)}.jpg",frame)
                        #print(n_frame_gpu)
            # Initialize cuda.stream object for stream synchronization
            #print(_time_f)
            #self.DATA.append((frame,_time_f))
            l_obj=frame_meta.obj_meta_list
            #print(l_obj,77777777777777777777777777777777)
            #src_=list(self.uri_dict.keys())[frame_meta.pad_index]
            #print(src_)
            #print("#"*100)

            if self.inferencer==1:
                dets=list()
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
                       # print(x,y,w,h,class_id,obj_id)
                        dets.append((x,y,w,h,class_id,obj_id))
                        #print(frame_meta.pad_index,x,y,w,h,class_id,obj_id)
                        #self.DATA.append((src_,frame,_time_f,(x,y,w,h,class_id,obj_id)))
                    except StopIteration:
                        break
                    try: 
                        l_obj=l_obj.next
                    except StopIteration:
                        break
                self.inferenced_data[stream_id]["data"].append((frame,dets))
                
                self.inferenced_data[stream_id]["num"]-=1
            else:
                pass
                #self.DATA.append((src_,frame,_time_f))   
            

      


            #print(len(self.DATA))

            stream = cp.cuda.stream.Stream(null=True) # Use null stream to prevent other cuda applications from making illegal memory access of buffer
            # Modify the red channel to add blue tint to image
            with stream:
                frame[:, :, 0] = 0.5 * frame[:, :, 0] + 0.5
            stream.synchronize()


            #print("Frame Number=", frame_number, "Number of Objects=",num_rects,"Vehicle_count=",obj_counter[PGIE_CLASS_ID_VEHICLE],"Person_count=",obj_counter[PGIE_CLASS_ID_PERSON])
            # Get frame rate through this probe
            
            #global perf_data
            #self.perf_data.update_fps(stream_index)
            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK
    
    #*****************************************************************************READ***********************************************************************************************
    def get_output(self):
        
        for stream_id in self.inferenced_data.copy():
            #print(self.inferenced_data[stream_id]["num"])
            if self.inferenced_data[stream_id]["num"]==0:
                #print(self.inferenced_data[stream_id]["num"])
                self.output_queue.append([stream_id,self.inferenced_data[stream_id]["data"]].copy())

                del self.inferenced_data[stream_id]
        return True