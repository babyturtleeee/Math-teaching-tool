import bpy
from bpy.types import Operator
import addon_utils

# Install these
import numpy as np

import scipy
from scipy.optimize import minimize
from scipy.misc import derivative

from sympy import diff, Symbol
from sympy.parsing.sympy_parser import parse_expr
from sympy import sin
from sympy import cos

def purge_orphans():
    if bpy.app.version >= (3, 0, 0):
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True
        )
    else:
        # call purge_orphans() recursively until there are no more orphan data blocks to purge
        result = bpy.ops.outliner.orphans_purge()
        if result.pop() != "CANCELLED":
            purge_orphans()

def recurLayerCollection(layerColl, collName):
    found = None
    if (layerColl.name == collName):
        return layerColl
    for layer in layerColl.children:
        found = recurLayerCollection(layer, collName)
        if found:
            return found

def enable_addon(addon_module_name):
    loaded_default, loaded_state = addon_utils.check(addon_module_name)
    if not loaded_state:
        addon_utils.enable(addon_module_name)

def update_prop(self, value):
    if self.is_dragging:
 
        variable_A = self.A
        variable_B = self.B
        variable_C = self.C
        x_size = self.x
        y_size = self.y
        x_start = self.start_x
        y_start = self.start_y
        func_string = self.function
        
        x_division = int(6.4 * x_size)
        y_division = int(6.4 * y_size)
        
        eqn = func_string.replace("A","{}").format(variable_A).replace("B","{}").format(variable_B).replace("C","{}").format(variable_C)
        
        func_string = eqn
        
        layer_collection = bpy.context.view_layer.layer_collection
        layerColl = recurLayerCollection(layer_collection, 'Surface')
        bpy.context.view_layer.active_layer_collection = layerColl
        
        C = bpy.context
        
        bpy.data.objects['Function_surface'].select_set(True) # Blender 2.8x
        bpy.ops.object.delete() 
        
        bpy.ops.mesh.primitive_z_function_surface(equation = eqn, div_x=x_division, div_y=y_division, size_x = x_size, size_y = y_size)
        
        new_surface = bpy.context.active_object 

        mat = bpy.data.materials.get("Material_height_map")
        
        if new_surface.data.materials:
            new_surface.data.materials[0] = mat
        else:
            new_surface.data.materials.append(mat)
        
        new_surface.name = "Function_surface"
        
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Function_wire_frame'].select_set(True) # Blender 2.8x
        bpy.ops.object.delete() 
        
        bpy.ops.mesh.primitive_z_function_surface(equation = eqn, div_x=x_division, div_y=y_division, size_x = x_size, size_y = y_size)
        new_surface_frame = bpy.context.active_object
        bpy.ops.object.modifier_add(type='WIREFRAME')
        
        mat_frame = bpy.data.materials.get("Material_wire_frame")
            
        if new_surface_frame.data.materials:
            new_surface_frame.data.materials[0] = mat_frame
        else:
            new_surface_frame.data.materials.append(mat_frame)
            
        new_surface_frame.name = "Function_wire_frame"
        
        bpy.ops.object.select_all(action='DESELECT')
        
        C.scene.objects.get("x_axis").location.x = -(x_size/2.0) 
        C.scene.objects.get("x_axis").location.y =  (y_size/2.0) 
        C.scene.objects.get("x_axis").dimensions[2] = (x_size) 
        
        C.scene.objects.get("x_arrow_head").location.x =  (x_size/2.0)
        C.scene.objects.get("x_arrow_head").location.y =  (y_size/2.0)
        
        C.scene.objects.get("x_axis_label").location.x =  0.0
        C.scene.objects.get("x_axis_label").location.y =  (y_size/2.0)
        
        C.scene.objects.get("y_axis").location.x = -(x_size/2.0) 
        C.scene.objects.get("y_axis").location.y = (y_size/2.0) 
        C.scene.objects.get("y_axis").dimensions[2] = (y_size) 
        
        C.scene.objects.get("y_arrow_head").location.x = -(x_size/2.0)
        C.scene.objects.get("y_arrow_head").location.y = -(y_size/2.0)
        
        C.scene.objects.get("y_axis_label").location.x = -(x_size/2.0)
        C.scene.objects.get("y_axis_label").location.y =  0.0
        
        C.scene.objects.get("z_axis").location.x = -(x_size/2.0) 
        C.scene.objects.get("z_axis").location.y =  (y_size/2.0) 
        C.scene.objects.get("z_axis").dimensions[2] = (variable_A*3.0) 
        
        C.scene.objects.get("z_axis_label").location.x = -(x_size/2.0) - 0.5
        C.scene.objects.get("z_axis_label").location.y =  (y_size/2.0) + 0.5
        C.scene.objects.get("z_axis_label").location.z =  (variable_A*3.0) / 2.0
        
        C.scene.objects.get("Slicing_1").location.y =  self.slice_1
        C.scene.objects.get("Slicing_2").location.x =  self.slice_2
        
        user_str = eqn
        
        x = np.arange(-5,5,0.05)
        y = np.arange(-5,5,0.05)
        
        starting_pos = (x_start, y_start, f(x_start, y_start,user_str))
        current_pos = starting_pos

        learning_rate = self.r
        
        trace = []
        trace.append(current_pos)
        
        for i in range(50):
            X_derivative, Y_derivative = partial_derivative(f, 0, [current_pos[0],current_pos[1],user_str]), partial_derivative(f, 1, [current_pos[0],current_pos[1],user_str])
            X_new, Y_new = current_pos[0] - learning_rate * X_derivative, current_pos[1] - learning_rate * Y_derivative
            current_pos = X_new, Y_new, f(X_new, Y_new,user_str)
            trace.append(current_pos)
        
        context = bpy.context
        
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, enter_editmode=False, location=(0.0,0.0,0.0))

        sphere = context.object 
        
        # making curve
        crv = bpy.data.curves.new('crv', 'CURVE')
        crv.dimensions = '3D'
        crv.bevel_depth = 0.01
        
        # Animate the eval time (keyframes)
        crv.path_duration = 110
        crv.eval_time = 0
        crv.keyframe_insert("eval_time", frame=10)
        crv.eval_time = 60
        crv.keyframe_insert("eval_time", frame=60)
        crv.eval_time = 110
        crv.keyframe_insert("eval_time", frame=110)
        # initialize spline
        spline = crv.splines.new(type='NURBS')
        spline.use_endpoint_u = True
        spline.use_endpoint_v = True
        # a spline for each point
        spline.points.add(len(trace)-1) # 1 point by default

        # assign the point coordinates to the spline points
        for p, new_co in zip(spline.points, trace):
            p.co = (new_co + tuple([1.0])) # (add nurbs weight)
            
        # make a new object with the curve
        obj = bpy.data.objects.new('Path', crv)
        
        context.scene.collection.objects.link(obj)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        sphere.select_set(True)
        sphere.parent = bpy.context.scene.objects['Path'] 
        bpy.ops.object.parent_set(type='FOLLOW')
        
        purge_orphans()
        
    else:
        print("Beginning dragging the slider !")
        self.is_dragging = True
        bpy.ops.draggableprop.input('INVOKE_DEFAULT')

def f(x, y, func): return eval(func)

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

def find_min_max_on_surface(obj_name):
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        print("Object '{}' not found in scene.".format(obj_name))
        return None, None

    min_z = min(obj.data.vertices, key=lambda v: v.co.z).co
    max_z = max(obj.data.vertices, key=lambda v: v.co.z).co

    return min_z, max_z

def extrude_surface_down(surface_obj, z):
    # Duplicate the surface object
    duplicated_obj = surface_obj.copy()
    duplicated_obj.data = surface_obj.data.copy()
    duplicated_obj.animation_data_clear()
    bpy.context.collection.objects.link(duplicated_obj)

    # Extrude the duplicated surface down to the specified z-coordinate
    bpy.context.view_layer.objects.active = duplicated_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, z)})
    bpy.ops.object.mode_set(mode='OBJECT')

    print("Duplicated surface extruded down to z = {}.".format(z))

class Maxmin_operator(Operator):
    bl_idname = "object.maxmin"
    bl_label = "Find max and min"

    def execute(self, context):
        min_value, max_value = find_min_max_on_surface("Function_surface")
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, enter_editmode=False, location=min_value)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, enter_editmode=False, location=max_value)
        print("Minimum Value on Surface:", min_value)
        print("Maximum Value on Surface:", max_value)
    
        return {'FINISHED'}
    
class Extrusion_operator(Operator):
    """ tooltip goes here """
    bl_idname = "object.output"
    bl_label = "Generate 3D Printing File"

    def execute(self, context):
        
        C = bpy.context
        thickness = bpy.context.scene.my_prop.thickness
        rel_pos_text = bpy.context.scene.my_prop.text_height
        scale = bpy.context.scene.my_prop.text_scale
        
        extrude_surface_down(bpy.data.objects.get("Function_surface"), -thickness*1.5)
    
        #Text object
        bpy.ops.object.text_add(location=(C.scene.objects.get("Function_surface").dimensions.x*rel_pos_text+0.01, 0, -thickness/2.0)) #Location is to be changed according to the position of the "wave mesh object".
        text_object = bpy.context.object
        text_object.data.align_x = 'CENTER'
        text_object.data.align_y = 'CENTER'
        text_object.data.body = "f(x) = " + bpy.context.scene.my_prop.function.replace("**", "^") #Type in the equation
        text_object.data.size = scale
              
        #Rotate 90 degrees on X axis and Z axis (change accordingly)
        text_object.rotation_euler[0] = 1.5708  # 90 degrees in radians
        text_object.rotation_euler[2] = 1.5708  # 90 degrees in radians
        
        #Mesh Conversion
        bpy.ops.object.convert(target='MESH')
        solidify_modifier = text_object.modifiers.new(name="Solidify", type='SOLIDIFY')
        solidify_modifier.thickness = 0.1  # CHANGE ACCORDINGLY
        
        #Smooth and Decimate 
        text_object.data.use_auto_smooth = True
        decimate_modifier = text_object.modifiers.new(name="Decimate", type='DECIMATE')
        decimate_modifier.ratio = 0.5  #Change this according to the result
        
        # Add a plane object
        bpy.ops.mesh.primitive_plane_add(size=20) #Change if you are modifying the wave plane
        plane_object = bpy.context.object

        # Position the plane above the cube
        plane_object.location.z = -thickness

        #Boolean
        o = bpy.data.objects.get("Function_surface.001");
        bool_modifier = o.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_modifier.object = text_object 
        bool_modifier.operation = 'DIFFERENCE'
        
        bool_modifier = o.modifiers.new(name="Boolean", type='BOOLEAN')
        bool_modifier.object = plane_object
        bool_modifier.operation = 'DIFFERENCE'
        
        return {'FINISHED'}

class DRAGGABLEPROP_input(bpy.types.Operator):
    bl_idname  = "draggableprop.input"
    bl_label   = ""
    stop: bpy.props.BoolProperty() 

    def modal(self, context, event):
        if self.stop:
            context.scene.my_prop.is_dragging = False
            print("End Dragging !")
            return {'FINISHED'}
        if event.value == 'RELEASE':
            self.stop = True

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        self.stop = False
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class PT_draggable_prop(bpy.types.Panel):
    bl_label = "Project Tool Main Panel"
    bl_idname = "PT_MainPanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Project Tool"

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        row.label(text= "Input function: f(x,y); Control Parameters: A, B and C")
        
        layout.prop(context.scene.my_prop, "function")
        layout.row().separator()
        
        box = layout.box()
        row = box.row()
        row.label(text= "Variables")
        row = box.row()
        row.prop(context.scene.my_prop, "A")
        row.prop(context.scene.my_prop, "B")
        row.prop(context.scene.my_prop, "C")
        layout.row().separator()
        
        box = layout.box()
        row = box.row()
        row.label(text= "Find Max Min")
        row = box.row()
        row.operator(Maxmin_operator.bl_idname,text = "Find Maximum and Minimum")
        
        box = layout.box()
        row = box.row()
        row.label(text= "Gradient Descent")
        row = box.row()
        row.prop(context.scene.my_prop, "start_x")
        row.prop(context.scene.my_prop, "start_y")
        row = box.row()
        row.prop(context.scene.my_prop, "r")
        layout.row().separator()
        
        row = layout.row()
        row.label(text= "Rednering Scene Adjustment")
        
        box = layout.box()
        row = box.row()
        row.label(text= "Domain Range")
        row = box.row()
        row.prop(context.scene.my_prop, "x")
        row.prop(context.scene.my_prop, "y")
        
        box = layout.box()
        row = box.row()
        row.label(text= "Function Slicing")
        row = box.row()
        row.prop(context.scene.my_prop, "slice_2")
        row.prop(context.scene.my_prop, "slice_1")
        layout.row().separator()
        
        layout.label(text= "Output Printing File")
        box = layout.box()
        row = box.row()
        row.label(text= "Thickness and Label Position")
        row = box.row()
        row.prop(context.scene.my_prop, "thickness")
        row = box.row()
        row.prop(context.scene.my_prop, "text_height")
        row.prop(context.scene.my_prop, "text_scale")
        row = box.row()
        row.operator(Extrusion_operator.bl_idname,text = "Generate")
       
def PanelRegistration():
    bpy.utils.register_class(DRAGGABLEPROP_input)  
    bpy.utils.register_class(PT_draggable_prop) 
    bpy.utils.register_class(DraggableProp)  
    bpy.utils.register_class(Extrusion_operator)  
    bpy.utils.register_class(Maxmin_operator)  
    bpy.types.Scene.my_prop = bpy.props.PointerProperty(type=DraggableProp)  

class DraggableProp(bpy.types.PropertyGroup):
    A: bpy.props.FloatProperty(update=update_prop, min=-100, max=100)
    B: bpy.props.FloatProperty(update=update_prop, min=-100, max=100)
    C: bpy.props.FloatProperty(update=update_prop, min=-100, max=100)
    x: bpy.props.FloatProperty(update=update_prop, min=0, max=100,default = 1.0)
    y: bpy.props.FloatProperty(update=update_prop, min=0, max=100,default = 1.0)
    
    start_x: bpy.props.FloatProperty(name = "starting location (x)",update=update_prop, min=-100, max=100,default = 0.0)
    start_y: bpy.props.FloatProperty(name = "Starting location (y)", update=update_prop, min=-100, max=100,default = 0.0)
    
    slice_1: bpy.props.FloatProperty(name = "Slicing(Y)",update=update_prop, min=-100, max=100,default = 0.0)
    slice_2: bpy.props.FloatProperty(name = "Slicing(X)", update=update_prop, min=-100, max=100,default = 0.0)
    
    function: bpy.props.StringProperty(name = "Function: ", default = "Type in Functions to be drawn")
    
    r: bpy.props.FloatProperty(name = "Learning rate", update=update_prop, min=0, max=5,default = 0.1)
    
    thickness: bpy.props.FloatProperty(name = "Extrusion Thickness",update=update_prop, min=-0, max=20,default = 10.0)
    text_height: bpy.props.FloatProperty(name = "Text Position (relative z)",update=update_prop, min=-0, max=5,default = 0.5)
    text_scale: bpy.props.FloatProperty(name = "Text Scale", update=update_prop, min=-0, max=5,default = 1.5)
    
    is_dragging: bpy.props.BoolProperty()

def main():
    purge_orphans()
    enable_addon(addon_module_name="add_mesh_extra_objects")
    enable_addon(addon_module_name="blender_colormaps")
    PanelRegistration()

if __name__ == "__main__":
    main()