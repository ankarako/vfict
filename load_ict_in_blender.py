"""
Blender addon to load ICT mesh with transferred neck LBS weights
Installation: Edit > Preferences > Add-ons > Install > Select this file
"""

bl_info = {
    "name": "ICT Neck LBS Loader",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > ICT LBS",
    "description": "Load ICT mesh with neck skinning weights",
    "category": "Import-Export",
}

import bpy
import os
import numpy as np
from bpy.props import StringProperty
from bpy.types import Operator, Panel, PropertyGroup


class ICTLBSProperties(PropertyGroup):
    """Properties for ICT LBS loader"""
    output_dir: StringProperty(
        name="Output Directory",
        description="Directory containing ICT mesh and LBS data",
        default=".output",
        subtype='DIR_PATH'
    )


class ICT_OT_LoadLBS(Operator):
    """Load ICT mesh with neck LBS weights"""
    bl_idname = "ict.load_lbs"
    bl_label = "Load ICT with Neck LBS"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        props = context.scene.ict_lbs_props
        output_dir = bpy.path.abspath(props.output_dir)

        # Paths
        ict_mesh_path = os.path.join(output_dir, "ict_filtered.obj")
        lbs_path = os.path.join(output_dir, "ict_neck_lbs.npz")

        # Check files exist
        if not os.path.exists(ict_mesh_path):
            self.report({'ERROR'}, f"ICT mesh not found at {ict_mesh_path}")
            return {'CANCELLED'}
        if not os.path.exists(lbs_path):
            self.report({'ERROR'}, f"LBS data not found at {lbs_path}")
            return {'CANCELLED'}

        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # Load mesh
        self.report({'INFO'}, f"Loading ICT mesh from {ict_mesh_path}")
        bpy.ops.wm.obj_import(filepath=ict_mesh_path)
        mesh_obj = bpy.context.selected_objects[0]
        mesh_obj.name = "ICT_Mesh"

        # Load LBS data
        self.report({'INFO'}, f"Loading neck LBS data from {lbs_path}")
        neck_weights, neck_joint = self.load_neck_lbs(lbs_path)

        # Create armature
        self.report({'INFO'}, "Creating armature with neck bone")
        armature_obj = self.create_armature_with_neck(neck_joint)

        # Apply weights
        self.report({'INFO'}, "Applying vertex weights")
        self.apply_vertex_weights(mesh_obj, neck_weights)

        # Bind mesh to armature
        self.report({'INFO'}, "Binding mesh to armature")
        self.bind_mesh_to_armature(mesh_obj, armature_obj)

        # Setup camera and lighting
        self.setup_camera_and_lighting()

        # Visualize weights
        self.visualize_weights(mesh_obj)

        self.report({'INFO'}, "ICT mesh loaded successfully! Switch to Pose Mode to test animation.")
        return {'FINISHED'}

    def load_neck_lbs(self, lbs_path):
        """Load neck weights and joint from NPZ file"""
        data = np.load(lbs_path)
        neck_weights = data['neck_weights']
        neck_joint = data['neck_joint']
        return neck_weights, neck_joint

    def create_armature_with_neck(self, neck_joint_pos):
        """Create armature with neck bone"""
        # Create armature
        armature = bpy.data.armatures.new("ICT_Armature")
        armature_obj = bpy.data.objects.new("ICT_Armature", armature)
        bpy.context.collection.objects.link(armature_obj)

        # Set armature display settings for better visibility
        armature.display_type = 'OCTAHEDRAL'
        armature.show_names = True
        armature_obj.show_in_front = True

        # Select and enter edit mode
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='EDIT')

        # Create neck bone (make it longer and more visible)
        neck_bone = armature.edit_bones.new("Neck")

        # Set bone positions (head below joint, tail at/above joint)
        bone_length = 0.3  # Make bone longer for visibility
        neck_bone.head = neck_joint_pos - np.array([0, 0, bone_length])
        neck_bone.tail = neck_joint_pos

        print(f"Neck bone created:")
        print(f"  Head: {neck_bone.head}")
        print(f"  Tail: {neck_bone.tail}")
        print(f"  Length: {neck_bone.length}")

        # Exit edit mode
        bpy.ops.object.mode_set(mode='OBJECT')

        return armature_obj

    def apply_vertex_weights(self, mesh_obj, neck_weights):
        """Apply neck weights to mesh vertices"""
        # Create vertex group for neck
        vgroup = mesh_obj.vertex_groups.new(name="Neck")

        # Assign weights to vertices
        for i, weight in enumerate(neck_weights):
            if weight > 1e-6:  # Only assign non-zero weights
                vgroup.add([i], weight, 'REPLACE')

        print(f"Applied weights to {len(neck_weights)} vertices")
        print(f"Weight range: [{neck_weights.min():.4f}, {neck_weights.max():.4f}]")
        print(f"Non-zero weights: {(neck_weights > 1e-6).sum()}")

    def bind_mesh_to_armature(self, mesh_obj, armature_obj):
        """Bind mesh to armature"""
        # Select mesh and armature
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        armature_obj.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj

        # Parent with armature deform
        bpy.ops.object.parent_set(type='ARMATURE_NAME')

        # Add armature modifier
        modifier = mesh_obj.modifiers.new(name="Armature", type='ARMATURE')
        modifier.object = armature_obj

    def visualize_weights(self, mesh_obj):
        """Enable weight paint visualization"""
        bpy.context.view_layer.objects.active = mesh_obj
        mesh_obj.select_set(True)

        # Set active vertex group
        if "Neck" in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.active_index = mesh_obj.vertex_groups["Neck"].index

        # Stay in object mode (user can switch manually)
        bpy.ops.object.mode_set(mode='OBJECT')

    def setup_camera_and_lighting(self):
        """Add basic camera and lighting"""
        # Add camera
        bpy.ops.object.camera_add(location=(0, -2, 0.5))
        camera = bpy.context.object
        camera.rotation_euler = (np.pi/2, 0, 0)
        bpy.context.scene.camera = camera

        # Add light
        bpy.ops.object.light_add(type='SUN', location=(0, 0, 5))
        light = bpy.context.object
        light.data.energy = 2.0


class ICT_PT_LoadPanel(Panel):
    """Panel for ICT LBS loader"""
    bl_label = "ICT Neck LBS Loader"
    bl_idname = "ICT_PT_load_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ICT LBS'

    def draw(self, context):
        layout = self.layout
        props = context.scene.ict_lbs_props

        layout.label(text="Load ICT mesh with Neck weights:")
        layout.prop(props, "output_dir")
        layout.operator("ict.load_lbs", text="Load ICT Mesh", icon='IMPORT')

        layout.separator()
        layout.label(text="After loading:")
        layout.label(text="1. Select armature (ICT_Armature)")
        layout.label(text="2. Enter Pose Mode (Ctrl+Tab)")
        layout.label(text="3. Select Neck bone")
        layout.label(text="4. Rotate (R key) to test")


# Registration
classes = (
    ICTLBSProperties,
    ICT_OT_LoadLBS,
    ICT_PT_LoadPanel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.ict_lbs_props = bpy.props.PointerProperty(type=ICTLBSProperties)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.ict_lbs_props


if __name__ == "__main__":
    register()
