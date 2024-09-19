import matplotlib.pyplot as plt
import numpy as np
from numba import jit

trjFile = 'prod_aligned.dat'
topFile = 'init.top'
sel_file = 'origami_indices.txt'

dr = 0.5        # shell thickness in nm
r_max = 25      # Maximum shell radius in nm

############# FUNCTIONS #############

def read_trjFile(trjFile):
    with open(trjFile) as f:
        content = f.readlines()
    return content

def extract_frame_data(content, start_idx):
    frame_data = []
    idx = start_idx
    MDstep, box, energy = None, None, None
    while idx < len(content):
        if content[idx].startswith("t ="):
            MDstep = content[idx]
        elif content[idx].startswith("b ="):
            box = content[idx]
        elif content[idx].startswith("E ="):
            energy = content[idx]
        else:
            while idx < len(content) and not content[idx].startswith("t =") and not content[idx].startswith("b =") and not content[idx].startswith("E ="):
                frame_data.append(content[idx].strip())
                idx += 1
            break
        idx += 1
    return MDstep, box, energy, frame_data, idx

def parse_frame(frame_data, ids):
    if isinstance(ids[0], list):
        ids_flat = [int(item) for sublist in ids for item in sublist]
        coords = [frame_data[i].split()[:3] for i in ids_flat]
    else:
        coords = [frame_data[i].split()[:3] for i in ids]
    return np.array(coords, dtype=float)

def read_selection_ids(sel_file):
    with open(sel_file) as f:
        ids = f.readlines()
    split_ids = [line.strip().split(',') for line in ids] ## edit the delimeter as needed!
    return split_ids

def read_topFile(topFile):
    """Reads the topology file and returns the number of bases, strands, and the topology data."""
    with open(topFile) as f:
        content = f.readlines()
    split_content = [line.split() for line in content]
    nbases = split_content[0][0]
    nstrands = split_content[0][1]
    topology = split_content[1:]
    return nbases, nstrands, topology

@jit(nopython=True)
def compute_min_distances(brush_coords, core_coords):
    num_brush = brush_coords.shape[0]
    num_core = core_coords.shape[0]
    min_distances = np.full(num_brush, np.inf)

    for i in range(num_brush):
        b = brush_coords[i]
        for j in range(num_core):
            c = core_coords[j]
            distance = np.linalg.norm(b - c)
            if distance < min_distances[i]:
                min_distances[i] = distance

    return min_distances

def convert_oxDNA_to_nms(lengths):
    factor = 0.8518
    if isinstance(lengths, list):
        converted_lengths = [x * factor for x in lengths]
    else:
        converted_lengths = factor * lengths
    return converted_lengths

def compute_rdf(distances, dr, r_max, nframes, deposited=False):
    """ computes the radial distribution function """
    """ for spherical shells of thickness dr [nm] """
    """ and a maximum shell radius r_max [nm] """

    num_bins = int((r_max - dr) / dr) + 1
    r_bins = np.linspace(dr, r_max, num_bins)

    if deposited == True:
        dV_bins = [0] * len(r_bins)
        for bin in range(len(r_bins)):
            dV_bins[bin] = 2*np.pi*r_bins[bin]**2*dr
    else:
        dV_bins = [0] * len(r_bins)
        for bin in range(len(r_bins)):
            dV_bins[bin] = 4*np.pi*r_bins[bin]**2*dr

    counts = [0] * len(r_bins)
    for d in all_min_distances:
        bin_index = np.searchsorted(r_bins, d, side='right')
        counts[bin_index] += 1

    rdf = [count / nframes / volume if volume > 0 else 0 for count, volume in zip(counts, dV_bins)]

    return rdf, r_bins

def plot_rdf(rdf, r_bins):
    plt.figure(figsize=(4, 3))
    plt.xlabel(r'$r [nm]$', fontsize=12)
    plt.ylabel(r'$ρ(r) [nuc/nm^3]$', fontsize=12)
    plt.gca().tick_params(axis='both', direction='in', width=1.0)
    plt.gca().spines['left'].set_linewidth(1.0)
    plt.gca().spines['right'].set_linewidth(1.0)
    plt.gca().spines['bottom'].set_linewidth(1.0)
    plt.gca().spines['top'].set_linewidth(1.0)
    plt.plot(r_bins,rdf,linewidth=1.5, color='k', label='$Disk_{50nm}BD_{50nuc}$')
    plt.legend()
    plt.savefig('rdf.png', dpi=300)

############# MAIN CODE #############

trjFile = 'prod_aligned.dat'
content = read_trjFile(trjFile)

topFile = 'init.top'
nbases, nstrands, top = read_topFile(topFile)

sel_file = 'origami_indices.txt'
core_ids = read_selection_ids(sel_file)

# find the indices
indices = np.arange(int(nbases))
flattened_core_ids = [item for sublist in core_ids for item in sublist]
core_ids_int = list(map(int, flattened_core_ids))
brush_ids = [id for id in indices if id not in core_ids_int]

#len(core_ids_int) + len(brush_ids) == int(nbases)

# avoid going unto the loop again when rerunning the code if frame_min_distance exists
# if it doesnt exist or it is empty, go in
frame_start_idx = 0
frame_min_distance = []

while frame_start_idx < len(content):
    if content[frame_start_idx].startswith("t ="):
        MDstep, box, energy, frame_data, frame_start_idx = extract_frame_data(content, frame_start_idx + 1)

        core_coords = parse_frame(frame_data, core_ids)
        brush_coords = parse_frame(frame_data, brush_ids)

        min_distances = compute_min_distances(brush_coords, core_coords)

        frame_min_distance.append(min_distances)

        #print("processed frame")

    else:
        frame_start_idx += 1

print("read trajectory and computed distances")

nframes = len(frame_min_distance)
all_min_distances = [item for sublist in frame_min_distance for item in sublist]
all_min_distances_nm = convert_oxDNA_to_nms(all_min_distances)

print("perfomed necessary unit conversions")

rdf, r_bins  = compute_rdf(all_min_distances_nm, dr, r_max, nframes, deposited=True)

np.savetxt('rdf_data.txt', np.column_stack((r_bins, rdf)), header='r_bins\t rdf', comments='')

print("computed rdf and saved txt file in current working directory")

plot_rdf(rdf, r_bins, path)

print("plotted rdf and saved image in current working directory")
