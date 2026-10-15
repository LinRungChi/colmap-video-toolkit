"""Render a textured still-life orbit with Blender Workbench (fast, noise-free). Demo data only."""
import math, sys
import numpy as np
import bpy
from mathutils import Vector

out = sys.argv[-1]
rng = np.random.default_rng(0)

def texture(name, tint, size=512):
    """Multi-scale random blobs: lots of distinct corners for SIFT."""
    img = np.zeros((size, size))
    for s in (4, 8, 16, 32, 64):
        g = rng.random((size // s + 1, size // s + 1))
        img += np.kron(g, np.ones((s, s)))[:size, :size] / s ** 0.3
    img = (img - img.min()) / (img.max() - img.min())
    rgba = np.dstack([img * t + (1 - t) * 0.15 for t in tint] + [np.ones_like(img)]).astype(np.float32)
    bi = bpy.data.images.new(name, size, size)
    bi.pixels.foreach_set(rgba.ravel())
    m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
    tx = nt.nodes.new("ShaderNodeTexImage"); tx.image = bi; tx.interpolation = "Closest"
    nt.links.new(tx.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_WORKBENCH"
sh = sc.display.shading
sh.light = "STUDIO"; sh.color_type = "TEXTURE"; sh.show_shadows = True; sh.show_cavity = True
sc.render.resolution_x, sc.render.resolution_y = 640, 480

def add(ob, mat):
    ob.data.materials.append(mat)
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.uv.smart_project(); bpy.ops.object.mode_set(mode="OBJECT")

bpy.ops.mesh.primitive_plane_add(size=6); add(bpy.context.object, texture("floor", (0.9, 0.85, 0.75)))
for kind, loc, s, tint in [("cube", (0, 0, 0.35), 0.7, (1.0, 0.55, 0.3)), ("sphere", (0.9, 0.5, 0.3), 0.3, (0.4, 0.6, 1.0)),
                           ("cyl", (-0.8, 0.6, 0.4), 0.3, (0.5, 1.0, 0.5)), ("cube", (0.6, -0.8, 0.2), 0.4, (1.0, 1.0, 0.6))]:
    if kind == "cube": bpy.ops.mesh.primitive_cube_add(size=s, location=loc)
    elif kind == "sphere": bpy.ops.mesh.primitive_uv_sphere_add(radius=s, location=loc, segments=48, ring_count=24)
    else: bpy.ops.mesh.primitive_cylinder_add(radius=s, depth=0.8, location=loc, vertices=48)
    ob = bpy.context.object; ob.rotation_euler = (0, 0, float(rng.uniform(0, 3))); add(ob, texture(kind, tint, 256))

bpy.ops.object.camera_add(); cam = bpy.context.object; sc.camera = cam; cam.data.lens = 30
N = 150
for i in range(N):
    a = math.radians(-70 + 140 * i / (N - 1))
    r = 3.2 + 0.08 * math.sin(i * 0.37); z = 1.9 + 0.06 * math.sin(i * 0.53)
    cam.location = Vector((r * math.sin(a), -r * math.cos(a), z))
    tgt = Vector((0.05 * math.sin(i * 0.29), 0.05 * math.cos(i * 0.31), 0.3))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}/{i:04d}.png"
    bpy.ops.render.render(write_still=True)
