"""Interactive 3D laminate schematics and constituent property explanations."""
from collections import Counter, defaultdict

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sensitivity_analysis import MATERIAL, PARAMETERS


LAYOUTS = {
    "Homogeneous laminate": "A uniform resin core sandwiched between two copper foils.",
    "Glass fibre weave": "Two plies of interlaced glass yarns embedded in the resin, between outer copper foils.",
    "Dispersed fillers": "Particles dispersed through the resin illustrate a filled laminate between copper foils.",
    "Multilayer copper": "A central copper layer, routed tracks above and below it, and patterned outer copper connected by plated vias.",
}
LEGACY_LAYOUTS = {
    "Homogeneous baseline": "Homogeneous laminate",
    "Balanced": "Glass fibre weave",
    "Resin-rich": "Glass fibre weave",
    "Glass-dense": "Glass fibre weave",
    "Filled matrix": "Dispersed fillers",
    "Copper interfaces": "Multilayer copper",
    "Copper-asymmetric": "Multilayer copper",
}
COLORS = {"Glass fiber reinforcement": "#368575", "Resin matrix": "#e7c887",
          "Glass weave architecture": "#648d76", "Copper foil interfaces": "#c87d55",
          "Fillers and additives": "#7384b0"}
FACTORS = {
    "Glass fiber reinforcement": dict(
        properties=["E", "alpha_lam"],
        effect="Glass volume fraction and fibre orientation change how strongly the reinforcement constrains the resin. "
               "This changes the laminate's effective directional stiffness and thermal expansion. In the model, with "
               "Poisson's ratio held fixed, greater modulus increases the stress-index magnitude for the same net mismatch "
               "strain. Changing laminate CTE changes the thermal mismatch with copper.\n\n"
               "Lower laminate CTE does not automatically mean lower mismatch: the relevant quantity is its difference "
               "from copper's CTE.",
        parameter_note="Related model parameters: E and αlam. Use values for the relevant material direction.",
        evidence="Veldhuijzen van Zanten et al. (2018) combined PCB experiments with micromechanical modelling. "
                 "Across the constructions studied, effective Young's modulus ranged from 11–31 GPa and CTE from "
                 "10–28 ppm/K, depending on reinforcement and fibre volume fraction.",
        scope="These are study-specific ranges, not universal material bounds.",
        title="Veldhuijzen van Zanten et al. (2018) · Method to determine thermoelastic material properties of constituent "
              "and copper-patterned layers of multilayer printed circuit boards",
        source="https://doi.org/10.1007/s10854-017-8449-2",
        limit="The model uses effective material properties; it does not calculate them from glass volume fraction or resolve local fibre restraint."),
    "Resin matrix": dict(
        properties=["D", "Csat", "beta", "E", "alpha_lam"],
        effect="Resin chemistry and the cured network influence water mobility, equilibrium absorption and swelling. "
               "These affect D, Csat and β, changing the time-dependent hygroscopic contribution. Resin chemistry "
               "and conditioning can also change stiffness and thermal behaviour.\n\n"
               "The direction of these changes must be established experimentally. Crosslink density alone does not "
               "reliably predict diffusivity.",
        parameter_note="Related model parameters: D, Csat, β, E and αlam.",
        evidence="Toscano et al. (2016) studied DGEBA/DDS and DGEBF/DDS epoxies in water at 50°C and 80°C, "
                 "using gravimetry, dynamic mechanical analysis and photoelastic stress measurements. The more highly "
                 "crosslinked DGEBA system had higher diffusivity and equilibrium uptake. The two systems developed "
                 "comparable peak swelling stresses but different stress-development kinetics.",
        scope="These were neat-resin specimens. The findings support mechanisms rather than directly supplying "
              "properties for a glass-reinforced laminate or a resin-rich pocket.",
        title="Toscano et al. (2016) · Water diffusion and swelling stresses in highly crosslinked epoxy matrices",
        source="https://doi.org/10.1016/j.polymdegradstab.2016.09.004",
        limit="The constant-property model does not capture moisture-induced stiffness changes or local stress gradients around resin-rich regions."),
    "Glass weave architecture": dict(
        properties=["E", "nu", "alpha_lam"],
        effect="Warp/fill yarn arrangement and fibre crimp change the reinforcement in each direction. This produces "
               "directional differences in effective modulus, Poisson's ratio and CTE. Consequently, the calculated "
               "stress index can differ between the laminate's x and y directions when direction-specific inputs are used.\n\n"
               "The evidence supports directional thermoelastic properties. A specific weave should not be claimed "
               "to reduce diffusivity without supporting transport measurements.",
        parameter_note="Related model parameters: direction-specific E, ν and αlam.",
        evidence="Sottos, Ockers and Swindeman (1999) characterised woven glass/epoxy boards using optical microscopy, "
                 "tensile testing and thermomechanical analysis above and below the glass-transition temperature Tg. "
                 "They measured elastic moduli, Poisson's ratios and CTEs, compared them with analytical models, "
                 "and investigated the effect of fibre crimp.",
        scope="The study concerns woven glass/epoxy circuit-board substrates; its thermoelastic results do not establish a weave-dependent diffusion ranking.",
        title="Sottos, Ockers & Swindeman (1999) · Thermoelastic properties of plain weave composites for multilayer circuit board applications",
        source="https://doi.org/10.1115/1.2792659",
        limit="The effective-property model does not resolve local stresses around individual yarns or resin pockets."),
    "Copper foil interfaces": dict(
        properties=["alpha_cu"],
        effect="Copper coverage changes the surfaces available for moisture entry, affecting moisture distribution "
               "and subsequent swelling. This is primarily a diffusion-boundary-condition effect, rather than a change "
               "in the laminate's intrinsic diffusivity.\n\n"
               "Interface condition also affects adhesion and resistance to separation. This is a separate pathway "
               "to delamination that the current stress index does not calculate.",
        parameter_note="Related model input: copper CTE (αCu). Moisture-exposure geometry is a boundary-condition consideration.",
        evidence="Weide-Zaage, Horaud and Frémont (2005) compared PCB moisture measurements with finite-element "
                 "simulations. Copper layers acted as diffusion barriers, changing moisture distributions and drying behaviour.",
        title="Weide-Zaage, Horaud & Frémont (2005) · Moisture diffusion in printed circuit boards: Measurements and finite-element simulations",
        source="https://doi.org/10.1016/j.microrel.2005.07.077",
        additional_research=[dict(
            evidence="Chan and Yuen (2009) examined epoxy–copper specimens conditioned for up to 1,000 hours at "
                     "85°C/85% RH. Interfacial moisture measurements and fracture-surface analysis linked transport "
                     "along the interface and changes in its cuprous oxide region to adhesion degradation in that system.",
            title="Chan & Yuen (2009) · Study of interfacial moisture diffusion at epoxy/Cu interface",
            source="https://doi.org/10.1163/156856109X433982",
            scope="The interfacial findings apply to the tested epoxy/copper system; they are not a universal adhesion-degradation law.",
        )],
        limit="The uptake model assumes one exposed face and an impermeable copper interface. Partial copper coverage, adhesion degradation, "
              "copper load sharing and local interface stresses are outside the current stress index."),
    "Fillers and additives": dict(
        properties=["E", "alpha_lam", "D", "Csat"],
        effect="Filler chemistry, loading and dispersion can change stiffness, CTE and moisture transport. "
               "In suitable silica-filled epoxies, changes in CTE and water uptake alter the thermal and hygroscopic "
               "contributions, while increased stiffness raises the stress scale for a given net mismatch.\n\n"
               "These effects compete, so more filler does not guarantee a smaller combined stress index. "
               "Lower CTE is beneficial for thermal mismatch only when it brings the laminate CTE closer to copper's CTE.",
        parameter_note="Related model parameters: E, αlam, D and Csat. A change in β requires separate swelling evidence.",
        evidence="Dittanet, Pearson and Kongkachuichay (2017) tested approximately 20 nm silica particles in DGEBF "
                 "and cycloaliphatic epoxies. Increasing filler content reduced CTE, moisture uptake and diffusivity, "
                 "and increased modulus in those formulations. At 85°C/85% RH, DGEBF equilibrium mass uptake "
                 "decreased from approximately 1.25% without filler to 0.84% at 35 wt% silica.",
        scope="The findings concern those silica-filled systems, not every filler or additive. Mass-uptake percentages "
              "require a consistent density and concentration basis before use as Csat.",
        title="Dittanet, Pearson & Kongkachuichay (2017) · Thermo-mechanical behaviors and moisture absorption of silica nanoparticle reinforcement in epoxy resins",
        source="https://doi.org/10.1016/j.ijadhadh.2017.06.006",
        limit="The model requires measured effective properties for a filled formulation; it does not infer them from filler loading."),
}


class Mesh:
    """Batch disconnected solids into one Plotly mesh per constituent."""

    def __init__(self):
        self.vertices = []
        self.triangles = []
        self.size = 0

    def add(self, vertices, triangles):
        vertices = np.asarray(vertices, dtype=float)
        self.vertices.append(vertices)
        self.triangles.append(np.asarray(triangles, dtype=int) + self.size)
        self.size += len(vertices)

    def box(self, x0, x1, y0, y1, z0, z1):
        if x1 <= x0 or y1 <= y0:
            return
        self.add(
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)],
        )

    def clip_y(self, minimum):
        """Slice each solid at the same plane and close the exposed section."""
        result = Mesh()
        tolerance = 1e-10
        offset = 0
        for vertices, faces in zip(self.vertices, self.triangles):
            # Faces in the batched mesh use global indices; work per solid.
            faces = faces - offset
            offset += len(vertices)
            if vertices[:, 1].min() >= minimum - tolerance:
                result.add(vertices, faces)
                continue
            if vertices[:, 1].max() <= minimum + tolerance:
                continue
            points, triangles, lookup = [], [], {}

            def vertex_id(point):
                key = tuple(np.round(point, 10))
                if key not in lookup:
                    lookup[key] = len(points)
                    points.append(point)
                return lookup[key]

            for face in faces:
                polygon = vertices[face]
                clipped = []
                for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                    if a[1] >= minimum:
                        clipped.append(a)
                    if (a[1] >= minimum) != (b[1] >= minimum):
                        intersection = a + (b - a) * (minimum - a[1]) / (b[1] - a[1])
                        intersection[1] = minimum
                        clipped.append(intersection)
                indices = []
                for point in clipped:
                    index = vertex_id(point)
                    if not indices or indices[-1] != index:
                        indices.append(index)
                if len(indices) > 1 and indices[-1] == indices[0]:
                    indices.pop()
                for index in range(1, len(indices) - 1):
                    triangles.append((indices[0], indices[index], indices[index + 1]))

            edges = Counter(tuple(sorted(edge)) for a, b, c in triangles
                            for edge in ((a, b), (b, c), (c, a)))
            boundary = {edge for edge, count in edges.items() if count == 1}
            neighbours = defaultdict(list)
            for a, b in boundary:
                neighbours[a].append(b)
                neighbours[b].append(a)
            while boundary:
                start, current = next(iter(boundary))
                loop = [start]
                previous = start
                boundary.remove(tuple(sorted((start, current))))
                while current != start:
                    loop.append(current)
                    following = next(index for index in neighbours[current] if index != previous)
                    boundary.remove(tuple(sorted((current, following))))
                    previous, current = current, following
                triangles.extend(section_cap(np.asarray(points), loop))
            result.add(points, triangles)
        return result

    def route(self, path, z0, z1, width=.18, y_min=0.):
        """Sweep a flat copper ribbon with mitred bends and a true cutaway edge."""
        path = np.asarray(path, dtype=float)
        directions = np.diff(path, axis=0)
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        normals = np.column_stack([-directions[:, 1], directions[:, 0]])
        offsets = np.empty_like(path)
        offsets[0], offsets[-1] = normals[0] * width / 2, normals[-1] * width / 2
        for index in range(1, len(path) - 1):
            bisector = normals[index - 1] + normals[index]
            offsets[index] = bisector * (width / 2) / np.dot(bisector, normals[index])
        left, right = path + offsets, path - offsets
        for index in range(len(path) - 1):
            polygon = [left[index], right[index], right[index + 1], left[index + 1]]
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                if a[1] >= y_min:
                    clipped.append(a)
                if (a[1] >= y_min) != (b[1] >= y_min):
                    clipped.append(a + (b - a) * (y_min - a[1]) / (b[1] - a[1]))
            # Intersections at a corner can repeat a vertex; discard them.
            polygon = []
            for point in clipped:
                if not polygon or not np.allclose(point, polygon[-1]):
                    polygon.append(point)
            if len(polygon) > 1 and np.allclose(polygon[0], polygon[-1]):
                polygon.pop()
            if len(polygon) < 3:
                continue
            polygon = np.asarray(polygon)
            area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                          - polygon[:, 1] * np.roll(polygon[:, 0], -1))
            if abs(area) < 1e-10:
                continue
            count = len(polygon)
            vertices = np.vstack([np.column_stack([polygon, np.full(count, z)]) for z in (z0, z1)])
            triangles = []
            for corner in range(1, count - 1):
                triangles.extend([(0, corner + 1, corner), (count, count + corner, count + corner + 1)])
            for corner in range(count):
                nxt = (corner + 1) % count
                triangles.extend([(corner, nxt, nxt + count), (corner, nxt + count, corner + count)])
            self.add(vertices, triangles)

    def yarn(self, coordinate, height, position, along_x):
        """Elliptical yarn with a sinusoidal over/under centreline."""
        theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
        length, sides = len(coordinate), len(theta)
        a = np.broadcast_to(coordinate[:, None], (length, sides))
        b = np.broadcast_to(position + .23 * np.cos(theta), a.shape)
        z = height[:, None] + .065 * np.sin(theta)
        rings = np.stack((a, b, z) if along_x else (b, a, z), axis=-1)
        points = rings.reshape(-1, 3).tolist()
        triangles = []
        for row in range(length - 1):
            for col in range(sides):
                a, b = row * sides + col, row * sides + (col + 1) % sides
                triangles.extend([(a, b, b + sides), (a, b + sides, a + sides)])
        points.extend([rings[0].mean(axis=0), rings[-1].mean(axis=0)])
        for col in range(sides):
            nxt = (col + 1) % sides
            last = (length - 1) * sides
            triangles.extend([(length * sides, nxt, col),
                              (length * sides + 1, last + col, last + nxt)])
        self.add(points, triangles)

    def particle(self, center, radius):
        """Closed sphere, using single vertices at each pole."""
        theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        phi = np.linspace(0, np.pi, 9)[1:-1]
        points = [(0, 0, radius)]
        for p in phi:
            points.extend(zip(radius * np.sin(p) * np.cos(theta),
                              radius * np.sin(p) * np.sin(theta),
                              np.full(len(theta), radius * np.cos(p))))
        points.append((0, 0, -radius))
        triangles = []
        for col in range(12):
            nxt = (col + 1) % 12
            triangles.extend([(0, 1 + col, 1 + nxt),
                              (len(points) - 1, 1 + 72 + nxt, 1 + 72 + col)])
        for row in range(6):
            for col in range(12):
                a, b = 1 + row * 12 + col, 1 + row * 12 + (col + 1) % 12
                triangles.extend([(a, a + 12, b + 12), (a, b + 12, b)])
        self.add(np.asarray(points) + center, triangles)

    def barrel(self, x, y, z0, z1, outer=.24, inner=.13):
        """A hollow plated via or annular copper pad, rather than a solid pin."""
        theta = np.linspace(0, 2 * np.pi, 24, endpoint=False)
        points = []
        for radius, z in [(outer, z0), (outer, z1), (inner, z0), (inner, z1)]:
            points.extend(zip(x + radius * np.cos(theta), y + radius * np.sin(theta), np.full(24, z)))
        triangles = []
        for ring_a, ring_b in [(0, 1), (3, 2), (2, 0), (1, 3)]:
            for col in range(24):
                nxt = (col + 1) % 24
                a, b = ring_a * 24 + col, ring_a * 24 + nxt
                c, d = ring_b * 24 + col, ring_b * 24 + nxt
                triangles.extend([(a, b, d), (a, d, c)])
        self.add(points, triangles)

    def trace(self, name, color, opacity=1, group=None, flatshading=False):
        vertices, faces = np.vstack(self.vertices), np.vstack(self.triangles)
        return go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
            color=color, opacity=opacity, legendgroup=group or name, showlegend=True,
            hovertemplate=name + "<extra></extra>", flatshading=flatshading,
            lighting=dict(ambient=.65, diffuse=.8, specular=.25, roughness=.5),
            lightposition=dict(x=100, y=-150, z=200),
        )


def section_cap(points, loop):
    """Triangulate a section boundary, retaining collinear boundary vertices."""
    section = points[:, [0, 2]]
    polygon = section[loop]
    area = np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                  - polygon[:, 1] * np.roll(polygon[:, 0], -1))
    if area < 0:
        loop = loop[::-1]
    remaining = list(loop)
    triangles = []

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    while len(remaining) > 3:
        for position, current in enumerate(remaining):
            previous, following = remaining[position - 1], remaining[(position + 1) % len(remaining)]
            a, b, c = section[[previous, current, following]]
            if cross(b - a, c - b) <= 1e-12:
                continue
            other = section[[index for index in remaining if index not in (previous, current, following)]]
            inside = ((cross(b - a, other - a) >= -1e-12)
                      & (cross(c - b, other - b) >= -1e-12)
                      & (cross(a - c, other - c) >= -1e-12))
            if inside.any():
                continue
            triangles.append((previous, current, following))
            remaining.pop(position)
            break
        else:
            raise ValueError("Could not close the laminate cross section.")
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    return triangles


def add_weave(warp, fill, y_min):
    x_positions = np.linspace(.65, 9.35, 8)
    y_positions = np.linspace(.7, 6.3, 6)
    x = np.linspace(0, 10, 120)
    y = np.linspace(y_min, 7, 90)
    for ply in (.65, 1.75):
        for row, position in enumerate(y_positions):
            if position - .23 < y_min:
                continue
            height = ply + .12 * np.cos(np.pi * (x - x_positions[0]) / (x_positions[1] - x_positions[0]) + row * np.pi)
            warp.yarn(x, height, position, along_x=True)
        for col, position in enumerate(x_positions):
            height = ply - .12 * np.cos(np.pi * (y - y_positions[0]) / (y_positions[1] - y_positions[0]) + col * np.pi)
            fill.yarn(y, height, position, along_x=False)


def filler_particles():
    """Fixed, nonoverlapping particles keep camera and layout comparisons stable."""
    rng = np.random.default_rng(21)
    particles = []
    for _ in range(2000):
        radius = rng.uniform(.065, .125)
        center = rng.uniform([radius, radius, radius], [10 - radius, 7 - radius, 2.4 - radius])
        if all(np.linalg.norm(center - other) > radius + r + .04 for other, r in particles):
            particles.append((center, radius))
        if len(particles) == 180:
            break
    return particles


COPPER_VIAS = ((1.2, 1.25), (2.3, 3.), (5.1, 5.7), (8.65, 4.9), (7.3, 1.15), (6.65, 3.4))
COPPER_ROUTES = {
    "top": [
        [(1.2, 1.25), (1.2, 2.1), (2.1, 3.), (2.3, 3.)],
        [(2.3, 3.), (3.35, 3.), (3.8, 3.45), (3.8, 4.7), (4.8, 5.7), (5.1, 5.7)],
        [(8.65, 4.9), (8.65, 3.95), (8.1, 3.4), (6.65, 3.4)],
        [(5.1, 5.7), (5.1, 6.3), (8.6, 6.3), (9.25, 5.65), (10., 5.65)],
    ],
    "inner_lower": [
        [(2.3, 3.), (2.3, 3.55), (2.85, 4.1), (4., 4.1), (4.35, 4.45),
         (4.35, 5.05), (4.7, 5.4), (5.7, 5.4), (6.05, 5.05), (6.05, 4.2),
         (6.5, 3.75), (7.5, 3.75), (8.65, 4.9)],
        [(1.2, 1.25), (2.1, 1.25), (2.6, 1.75), (4.5, 1.75), (4.95, 2.2),
         (4.95, 2.9), (5.45, 3.4), (6.65, 3.4)],
        [(5.1, 5.7), (4.4, 6.4), (.85, 6.4), (.5, 6.05), (.5, 4.7), (1.1, 4.1), (0., 4.1)],
    ],
    "inner_upper": [
        [(2.3, 3.), (1.45, 3.), (.85, 3.6), (.85, 5.), (1.45, 5.6), (3.2, 5.6),
         (3.65, 6.05), (4.75, 6.05), (5.1, 5.7)],
        [(5.1, 5.7), (5.1, 4.95), (5.6, 4.45), (5.6, 3.95), (6.15, 3.4), (6.65, 3.4)],
        [(8.65, 4.9), (9.15, 4.4), (9.15, 2.1), (8.2, 1.15), (7.3, 1.15)],
        [(1.2, 1.25), (1.2, .65), (3.4, .65), (3.95, 1.2), (3.95, 2.6), (4.45, 3.1),
         (4.9, 3.1), (5.3, 2.7), (5.3, 0.)],
    ],
    "bottom": [
        [(1.2, 1.25), (1.85, 1.9), (3., 1.9), (3.55, 1.35), (4.4, 1.35),
         (4.95, 1.9), (6.55, 1.9), (7.3, 1.15)],
        [(2.3, 3.), (2.3, 4.5), (3.2, 5.4), (6.15, 5.4), (6.75, 6.),
         (8., 6.), (8.65, 5.35), (8.65, 4.9)],
        [(6.65, 3.4), (7.6, 3.4), (8.1, 2.9), (9.5, 2.9), (10., 2.9)],
        [(5.1, 5.7), (5.1, 6.35), (2., 6.35), (1.25, 5.6), (0., 5.6)],
    ],
}


def add_copper_routes(mesh, layer, z0, z1, y_min):
    for route in COPPER_ROUTES[layer]:
        path = np.array(route, dtype=float)
        # Stop at the annular pads so the trace does not fill a via's bore.
        for endpoint, neighbour in ((0, 1), (-1, -2)):
            if any(np.allclose(path[endpoint], center) for center in COPPER_VIAS):
                direction = path[neighbour] - path[endpoint]
                path[endpoint] += .15 * direction / np.linalg.norm(direction)
        mesh.route(path, z0, z1, y_min=y_min)


def structure_figure(layout, focus="All constituents", view="Cross section", resin_opacity=.22):
    """Unitless 3D geometry; visual controls never modify material calculations."""
    layout = LEGACY_LAYOUTS.get(layout, layout)
    if layout not in LAYOUTS:
        raise ValueError("Choose a supported laminate structure.")
    if view == "Cutaway":
        view = "Cross section"
    if view not in ("Cross section", "See-through", "Lift outer foils"):
        raise ValueError("Choose a supported 3D view.")
    cross_section = view == "Cross section"
    y_min = 2.2 if cross_section else 0.
    gap = .8 if view == "Lift outer foils" else 0.
    copper = "Copper foil interfaces"
    resin = Mesh()
    resin.box(0, 10, 0, 7, 0, 2.4)
    bottom, top = Mesh(), Mesh()
    patterned = layout == "Multilayer copper"
    if patterned:
        add_copper_routes(top, "top", 2.4 + gap, 2.52 + gap, 0.)
        add_copper_routes(bottom, "bottom", -.12 - gap, -gap, 0.)
    else:
        top.box(0, 10, 0, 7, 2.4 + gap, 2.52 + gap)
        bottom.box(0, 10, 0, 7, -.12 - gap, -gap)

    figure = go.Figure()

    def add(mesh, label, color, factor, alpha=1):
        if cross_section:
            mesh = mesh.clip_y(y_min)
        if not mesh.vertices:
            return
        if focus not in ("All constituents", factor):
            alpha = min(alpha, .08)
        figure.add_trace(mesh.trace(label, color, alpha, factor,
                                    flatshading=factor in ("Resin matrix", copper)))

    # Add the translucent envelope last to make internal solids easier to see.
    foil_alpha = .22 if view == "See-through" else 1. if cross_section else .96
    add(bottom, "Bottom copper foil" if not patterned else "Bottom copper traces", COLORS[copper], copper, foil_alpha)
    add(top, "Top copper foil" if not patterned else "Top copper traces", "#df9b65", copper, foil_alpha)
    if layout == "Glass fibre weave":
        warp, fill = Mesh(), Mesh()
        add_weave(warp, fill, 0.)
        add(warp, "Glass yarns · warp", "#368575", "Glass fiber reinforcement")
        add(fill, "Glass yarns · fill", "#91b7a7", "Glass fiber reinforcement")
    elif layout == "Dispersed fillers":
        particles = Mesh()
        for center, radius in filler_particles():
            particles.particle(center, radius)
        add(particles, "Dispersed filler particles", COLORS["Fillers and additives"], "Fillers and additives")
    elif patterned:
        lower, middle, upper, vias, pads = Mesh(), Mesh(), Mesh(), Mesh(), Mesh()
        add_copper_routes(lower, "inner_lower", .77, .85, 0.)
        middle.box(0, 10, 0, 7, 1.16, 1.24)
        add_copper_routes(upper, "inner_upper", 1.55, 1.63, 0.)
        for x, y in COPPER_VIAS:
            vias.barrel(x, y, -.12, 2.52, outer=.20)
            for z0, z1 in ((-.12 - gap, -gap), (.77, .85), (1.16, 1.24),
                           (1.55, 1.63), (2.4 + gap, 2.52 + gap)):
                pads.barrel(x, y, z0, z1, outer=.34)
        add(lower, "Internal copper layer · lower routes", "#aa6843", copper)
        add(middle, "Central copper layer", "#bd7c50", copper)
        add(upper, "Internal copper layer · upper routes", "#e1a170", copper)
        add(vias, "Plated through-vias", "#ae653e", copper)
        add(pads, "Via pads", "#d58b58", copper)

    add(resin, "Homogeneous resin core" if layout == "Homogeneous laminate" else "Resin matrix",
        COLORS["Resin matrix"], "Resin matrix", resin_opacity)
    # A quiet outline gives the transparent resin a clearly readable volume.
    edge_x, edge_y, edge_z = [], [], []
    corners = [(0, y_min, 0), (10, y_min, 0), (10, 7, 0), (0, 7, 0),
               (0, y_min, 2.4), (10, y_min, 2.4), (10, 7, 2.4), (0, 7, 2.4)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        for index in (a, b):
            x, y, z = corners[index]
            edge_x.append(x)
            edge_y.append(y)
            edge_z.append(z)
        edge_x.append(None)
        edge_y.append(None)
        edge_z.append(None)
    figure.add_trace(go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode="lines",
                                 line=dict(color="#baaa82", width=2), hoverinfo="skip", showlegend=False))
    axis = dict(visible=False, showbackground=False, showspikes=False)
    figure.update_layout(
        height=560, margin=dict(l=0, r=0, t=65, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(xaxis=dict(axis, range=[-.5, 10.5]), yaxis=dict(axis, range=[-.5, 7.5]),
                   zaxis=dict(axis, range=[-1.1, 3.6]), bgcolor="rgba(0,0,0,0)",
                   aspectmode="data", dragmode="orbit",
                   camera=dict(eye=dict(x=1.45, y=-1.65, z=1.15), projection=dict(type="perspective")),
                   uirevision=f"laminate-{view}"),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11),
                    groupclick="toggleitem"),
    )
    return figure


PROPERTY_CARDS = {
    "Resin matrix": (
        "Resin matrix", "Moisture uptake · swelling",
        "The resin surrounds the reinforcement or particles. Its chemistry and cure influence how water moves "
        "through the laminate, how much is absorbed, and how much the matrix swells.",
        "D · Csat · β · E · αlam",
    ),
    "Glass fiber reinforcement": (
        "Glass fibre reinforcement", "Stiffness · thermal expansion",
        "Glass yarns constrain the resin. Fibre content and orientation change effective stiffness and thermal "
        "expansion; the properties depend on the direction being considered.",
        "E · αlam",
    ),
    "Glass weave architecture": (
        "Weave architecture", "Different behaviour along warp and fill",
        "Yarns pass over and under one another. Their spacing and crimp leave resin-rich pockets and change "
        "directional stiffness, Poisson’s ratio and thermal expansion.",
        "Directional E · ν · αlam",
    ),
    "Fillers and additives": (
        "Fillers and additives", "Stiffness · expansion · moisture transport",
        "Silica particles increased stiffness and reduced thermal expansion and moisture uptake in the cited epoxy "
        "systems. The effect depends on filler chemistry, loading and dispersion.",
        "E · αlam · D · Csat",
    ),
    "Copper foil interfaces": (
        "Patterned copper and vias", "Coverage · interfaces · local restraint",
        "Etched outer foils leave areas of resin exposed. Copper traces and plated vias introduce additional "
        "interfaces and local constraints. Copper coverage changes the paths available for moisture entry.",
        "Copper coverage · αCu · interface geometry",
    ),
    "Internal copper": (
        "Internal copper layers", "Central copper layer · routed tracks",
        "A continuous copper layer sits at the centre, with routed tracks above and below it. Free trace ends reach "
        "the board edges; via pads connect the layers. Copper content affects the board’s effective "
        "thermoelastic properties, while copper layers act as barriers that redistribute moisture.",
        "Layer sequence · copper fraction · transport paths",
    ),
}

LAYOUT_FACTORS = {
    "Homogeneous laminate": ["Resin matrix", "Copper foil interfaces"],
    "Glass fibre weave": ["Glass fiber reinforcement", "Glass weave architecture", "Resin matrix"],
    "Dispersed fillers": ["Fillers and additives", "Resin matrix"],
    "Multilayer copper": ["Copper foil interfaces", "Internal copper", "Resin matrix"],
}


def render_property_card(factor, homogeneous=False):
    title, subtitle, description, inputs = PROPERTY_CARDS[factor]
    source_factor = "Copper foil interfaces" if factor == "Internal copper" else factor
    note = FACTORS[source_factor]
    if homogeneous and factor == "Copper foil interfaces":
        title, subtitle = "Outer copper foils", "Thermal mismatch · moisture entry"
        description = ("Continuous copper foils bound the resin on both faces. Their expansion relative to the "
                       "laminate creates thermal mismatch. Copper coverage also changes moisture-entry paths.")
        inputs = "αlam − αCu · copper coverage"
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(subtitle)
        st.write(description)
        st.caption(inputs)
        if factor == "Internal copper":
            st.markdown(f"[Thermoelastic study]({FACTORS['Glass fiber reinforcement']['source']}) · "
                        f"[Moisture study]({note['source']})")
        else:
            st.markdown(f"[Research reference]({note['source']})")


def render_showcase():
    st.header("3D Laminate Visualiser")
    st.caption("Look inside a copper–laminate–copper stack and explore how reinforcement, fillers and copper features change its structure.")
    previous = st.session_state.get("inverse_structure_layout")
    if previous in LEGACY_LAYOUTS:
        st.session_state["inverse_structure_layout"] = LEGACY_LAYOUTS[previous]
    if st.session_state.get("inverse_structure_view") == "Cutaway":
        st.session_state["inverse_structure_view"] = "Cross section"
    viewer, details = st.columns([2.1, 1], gap="large")
    with viewer:
        layout = st.radio("Laminate structure", list(LAYOUTS), horizontal=True,
                          key="inverse_structure_layout", persist_state="session")
        st.write(LAYOUTS[layout])
        view_col, focus_col = st.columns(2)
        view = view_col.selectbox("View", ["Cross section", "See-through", "Lift outer foils"],
                                  key="inverse_structure_view", persist_state="session")
        focus_options = ["All constituents", "Resin matrix", "Copper foil interfaces"]
        if layout == "Glass fibre weave":
            focus_options.append("Glass fiber reinforcement")
        elif layout == "Dispersed fillers":
            focus_options.append("Fillers and additives")
        if st.session_state.get("inverse_structure_focus") not in focus_options:
            st.session_state.pop("inverse_structure_focus", None)
        focus = focus_col.selectbox("Highlight", focus_options, key="inverse_structure_focus")
        opacity = st.slider("Resin opacity", min_value=.05, max_value=.75, value=.22, step=.01,
                             key="inverse_structure_opacity", persist_state="session")
        st.plotly_chart(
            structure_figure(layout, focus, view, opacity), width="stretch", key="inverse_structure_figure",
            config=dict(displaylogo=False, scrollZoom=True, toImageButtonOptions=dict(filename="laminate_3d")),
        )
        st.caption("Drag to rotate · scroll to zoom · right-drag to pan · use the camera reset button to restore the view.")
        if view == "Cross section":
            st.caption("The entire stack is sliced on one plane, exposing the cross sections of the resin, copper and any reinforcement or fillers.")
        elif view == "Lift outer foils":
            st.caption("Outer copper is lifted away for inspection; internal features stay in their assembled positions.")
        st.caption("Schematic, not to scale. Fibre, particle and copper dimensions are exaggerated for visibility. "
                   "These visual controls do not change the material recommender’s inputs.")
    with details:
        st.markdown("#### How the structure changes properties")
        for factor in LAYOUT_FACTORS[layout]:
            render_property_card(factor, homogeneous=layout == "Homogeneous laminate")
        saved = st.session_state.get("sensitivity_result")
        if (saved and saved["signature"] == st.session_state.get("sensitivity_current_signature")
                and "ST" in saved["result"]["indices"]):
            top = saved["result"]["indices"]
            top_keys = top[top["Parameter"].isin(MATERIAL)].nlargest(3, "ST")["Parameter"].tolist()
            related = {key for factor in LAYOUT_FACTORS[layout]
                       for key in FACTORS.get(factor, {}).get("properties", [])}
            relevant = [PARAMETERS[key][0] for key in top_keys if key in related]
            if relevant:
                st.info("Related to your leading sensitivity parameters: " + ", ".join(relevant))
    with st.expander("Research and property details"):
        for factor in LAYOUT_FACTORS[layout]:
            if factor == "Internal copper":
                continue
            note = FACTORS[factor]
            title = "Outer copper foils" if layout == "Homogeneous laminate" and factor == "Copper foil interfaces" else PROPERTY_CARDS[factor][0]
            st.markdown(f"**{title}**")
            st.write(note["effect"])
            st.caption(note["parameter_note"])
            for study in [note, *note.get("additional_research", [])]:
                st.write(study["evidence"])
                st.markdown(f"[{study['title']}]({study['source']})")
