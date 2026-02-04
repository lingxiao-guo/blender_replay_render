"""
Import a scene PLY and object OBJ into Blender.

Usage (from command line):
  blender --python import_scene_objects.py -- <task_name>

Or open in Blender > Scripting > Run Script.
"""

import argparse
import os
import sys

import bpy

CLEAR_SCENE = False
SCENE_COLLECTION_NAME = "ImportedScene"
OBJECT_COLLECTION_NAME = "ImportedObjects"


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def get_new_objects(before_names):
    return [obj for name, obj in bpy.data.objects.items() if name not in before_names]


def ensure_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def move_objects_to_collection(objects, collection):
    for obj in objects:
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collection.objects.link(obj)


def import_ply(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.ply_import(filepath=path)
    return get_new_objects(before)


def import_obj(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=path)
    return get_new_objects(before)


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(
        description="Import a task scene PLY and all object OBJs/textures."
    )
    parser.add_argument(
        "task_name",
        help="Task folder name under this script directory (e.g. turning_on_radio).",
    )
    return parser.parse_args(argv)


def find_object_assets(task_dir):
    object_assets = []
    for root, _dirs, files in os.walk(task_dir):
        if "textured_mesh.obj" in files:
            obj_path = os.path.join(root, "textured_mesh.obj")
            tex_path = os.path.join(root, "material_0.png")
            if not os.path.exists(tex_path):
                raise FileNotFoundError(
                    f"Missing texture for object: {obj_path} (expected {tex_path})"
                )
            object_assets.append((obj_path, tex_path))
    object_assets.sort()
    if not object_assets:
        raise FileNotFoundError(
            f"No textured_mesh.obj files found under task directory: {task_dir}"
        )
    return object_assets


def material_name_from_path(image_path):
    parent = os.path.basename(os.path.dirname(image_path))
    if parent:
        return f"ObjectEmission_{parent}"
    return f"ObjectEmission_{os.path.splitext(os.path.basename(image_path))[0]}"


def create_emission_material(name, color_attribute="Col", strength=1.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission_node = nodes.new("ShaderNodeEmission")
    attr_node = nodes.new("ShaderNodeAttribute")
    attr_node.attribute_name = color_attribute
    emission_node.inputs["Strength"].default_value = strength

    links.new(attr_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], out_node.inputs["Surface"])
    return mat


def get_or_load_image(image_path):
    abs_path = os.path.abspath(image_path)
    for img in bpy.data.images:
        if bpy.path.abspath(img.filepath) == abs_path:
            return img
    return bpy.data.images.load(abs_path)


def create_emission_texture_material(name, image_path, strength=1.0):
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    image = get_or_load_image(image_path)

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out_node = nodes.new("ShaderNodeOutputMaterial")
    emission_node = nodes.new("ShaderNodeEmission")
    tex_node = nodes.new("ShaderNodeTexImage")
    tex_node.image = image
    emission_node.inputs["Strength"].default_value = strength

    links.new(tex_node.outputs["Color"], emission_node.inputs["Color"])
    links.new(emission_node.outputs["Emission"], out_node.inputs["Surface"])
    return mat


def assign_material(objects, material):
    for obj in objects:
        if obj.type != "MESH":
            continue
        if obj.data.materials:
            obj.data.materials[0] = material
        else:
            obj.data.materials.append(material)


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, args.task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    scene_ply_path = os.path.join(task_dir, "tetra_mesh_binary_search_7.ply")
    object_assets = find_object_assets(task_dir)

    if CLEAR_SCENE:
        clear_scene()

    scene_collection = ensure_collection(SCENE_COLLECTION_NAME)
    object_collection = ensure_collection(OBJECT_COLLECTION_NAME)

    scene_objs = import_ply(scene_ply_path)
    move_objects_to_collection(scene_objs, scene_collection)

    scene_mat = create_emission_material("SceneEmission")
    assign_material(scene_objs, scene_mat)

    material_cache = {}
    object_objs_all = []
    for obj_path, tex_path in object_assets:
        obj_objs = import_obj(obj_path)
        move_objects_to_collection(obj_objs, object_collection)
        if tex_path not in material_cache:
            material_cache[tex_path] = create_emission_texture_material(
                material_name_from_path(tex_path), tex_path
            )
        assign_material(obj_objs, material_cache[tex_path])
        object_objs_all.extend(obj_objs)

    bpy.context.view_layer.update()
    print(
        "Imported "
        f"{len(scene_objs)} scene object(s) and "
        f"{len(object_objs_all)} object(s) "
        f"from {len(object_assets)} object file(s)."
    )


if __name__ == "__main__":
    main()
