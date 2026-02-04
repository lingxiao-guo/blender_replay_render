"""
URDF Importer for Blender
Run this script inside Blender's Python environment:
  - Open Blender
  - Go to Scripting tab
  - Open this file and click "Run Script"

Or run from command line:
  blender --python import_urdf.py
"""

import bpy
import xml.etree.ElementTree as ET
import os
import math
from mathutils import Matrix, Vector, Euler

# Configuration - modify this path as needed
URDF_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/urdf/r1pro.urdf"
# When False, keep existing scene objects and just add the robot.
CLEAR_SCENE = False


def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def parse_origin(origin_elem):
    """Parse origin element and return translation and rotation."""
    xyz = [0, 0, 0]
    rpy = [0, 0, 0]

    if origin_elem is not None:
        xyz_str = origin_elem.get('xyz', '0 0 0')
        rpy_str = origin_elem.get('rpy', '0 0 0')
        xyz = [float(v) for v in xyz_str.split()]
        rpy = [float(v) for v in rpy_str.split()]

    return xyz, rpy


def create_transform_matrix(xyz, rpy):
    """Create a 4x4 transformation matrix from xyz translation and rpy rotation."""
    # RPY is roll (X), pitch (Y), yaw (Z) in radians
    rot_matrix = Euler((rpy[0], rpy[1], rpy[2]), 'XYZ').to_matrix().to_4x4()
    trans_matrix = Matrix.Translation(Vector(xyz))
    return trans_matrix @ rot_matrix


def import_mesh(mesh_path, link_name):
    """Import a mesh file and return the created object."""
    if not os.path.exists(mesh_path):
        print(f"Warning: Mesh file not found: {mesh_path}")
        # Create a placeholder cube
        bpy.ops.mesh.primitive_cube_add(size=0.05)
        obj = bpy.context.active_object
        obj.name = link_name
        return obj

    ext = os.path.splitext(mesh_path)[1].lower()

    # Store existing objects to find new one after import
    existing_objects = set(bpy.data.objects.keys())

    if ext == '.obj':
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif ext == '.stl':
        bpy.ops.wm.stl_import(filepath=mesh_path)
    elif ext == '.dae':
        bpy.ops.wm.collada_import(filepath=mesh_path)
    elif ext == '.ply':
        bpy.ops.wm.ply_import(filepath=mesh_path)
    else:
        print(f"Warning: Unsupported mesh format: {ext}")
        bpy.ops.mesh.primitive_cube_add(size=0.05)
        obj = bpy.context.active_object
        obj.name = link_name
        return obj

    # Find newly created objects
    new_objects = [obj for name, obj in bpy.data.objects.items()
                   if name not in existing_objects]

    if not new_objects:
        print(f"Warning: No objects imported from {mesh_path}")
        return None

    # If multiple objects, join them
    if len(new_objects) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in new_objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = new_objects[0]
        bpy.ops.object.join()
        obj = bpy.context.active_object
    else:
        obj = new_objects[0]

    obj.name = link_name
    return obj


def import_urdf(urdf_path, clear_scene_first=True):
    """Import a URDF file into Blender."""

    # Parse URDF
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    robot_name = root.get('name', 'robot')
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    print(f"Importing URDF: {robot_name}")

    # Clear scene (optional)
    if clear_scene_first:
        clear_scene()

    # Create root empty for the robot
    bpy.ops.object.empty_add(type='ARROWS')
    robot_root = bpy.context.active_object
    robot_root.name = robot_name

    # Dictionary to store link objects
    link_objects = {}

    # Parse all links
    for link_elem in root.findall('link'):
        link_name = link_elem.get('name')
        print(f"  Processing link: {link_name}")

        # Find visual geometry
        visual_elem = link_elem.find('visual')
        if visual_elem is not None:
            geometry_elem = visual_elem.find('geometry')
            mesh_elem = geometry_elem.find('mesh') if geometry_elem is not None else None

            if mesh_elem is not None:
                mesh_filename = mesh_elem.get('filename')
                mesh_path = os.path.join(urdf_dir, mesh_filename)

                # Import the mesh
                obj = import_mesh(mesh_path, link_name)

                if obj:
                    # Apply visual origin transform
                    origin_elem = visual_elem.find('origin')
                    xyz, rpy = parse_origin(origin_elem)

                    # Store original mesh transform (visual offset)
                    visual_matrix = create_transform_matrix(xyz, rpy)

                    # Apply transform to mesh data (not object)
                    obj.data.transform(visual_matrix)

                    # Reset object transform
                    obj.matrix_world = Matrix.Identity(4)

                    link_objects[link_name] = obj
            else:
                # No mesh, create empty
                bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.02)
                obj = bpy.context.active_object
                obj.name = link_name
                link_objects[link_name] = obj
        else:
            # No visual, create empty
            bpy.ops.object.empty_add(type='PLAIN_AXES', radius=0.02)
            obj = bpy.context.active_object
            obj.name = link_name
            link_objects[link_name] = obj

    # Parse all joints and set up hierarchy
    joint_info = {}
    for joint_elem in root.findall('joint'):
        joint_name = joint_elem.get('name')
        joint_type = joint_elem.get('type')

        parent_elem = joint_elem.find('parent')
        child_elem = joint_elem.find('child')

        if parent_elem is None or child_elem is None:
            continue

        parent_link = parent_elem.get('link')
        child_link = child_elem.get('link')

        origin_elem = joint_elem.find('origin')
        xyz, rpy = parse_origin(origin_elem)

        joint_info[joint_name] = {
            'type': joint_type,
            'parent': parent_link,
            'child': child_link,
            'xyz': xyz,
            'rpy': rpy
        }

    # Find root link (link with no parent joint)
    child_links = set(j['child'] for j in joint_info.values())
    parent_links = set(j['parent'] for j in joint_info.values())
    root_links = parent_links - child_links

    # Set up hierarchy
    for joint_name, info in joint_info.items():
        parent_link = info['parent']
        child_link = info['child']
        xyz = info['xyz']
        rpy = info['rpy']

        if child_link in link_objects and parent_link in link_objects:
            child_obj = link_objects[child_link]
            parent_obj = link_objects[parent_link]

            # Set parent
            child_obj.parent = parent_obj

            # Set local transform (joint origin)
            transform_matrix = create_transform_matrix(xyz, rpy)
            child_obj.matrix_parent_inverse = Matrix.Identity(4)
            child_obj.matrix_local = transform_matrix

    # Parent root links to robot root
    for link_name in root_links:
        if link_name in link_objects:
            link_objects[link_name].parent = robot_root

    # Also parent any orphan links
    for link_name, obj in link_objects.items():
        if obj.parent is None:
            obj.parent = robot_root

    # Update scene
    bpy.context.view_layer.update()

    # Frame all objects
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            override = {'area': area, 'region': area.regions[-1]}
            with bpy.context.temp_override(**override):
                bpy.ops.view3d.view_all()
            break

    print(f"URDF import complete: {len(link_objects)} links imported")
    return robot_root


if __name__ == "__main__":
    import_urdf(URDF_PATH, clear_scene_first=CLEAR_SCENE)
