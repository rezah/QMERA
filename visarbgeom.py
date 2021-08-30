from matplotlib import pyplot as plt


def clean():
    import gc
    plt.close('all')
    gc.collect()


def get_3d_pos(Lx, Ly, Lz, a=22, b=45, p=0.2):
    import math
    import itertools
    return {
        (i, j, k): (
            + i * math.cos(math.pi * a / 180) +
            j * math.cos(math.pi * b / 180) / 2**p,
            - i * math.sin(math.pi * a / 180) +
            j * math.sin(math.pi * b / 180) / 2**p + k
        )
        for i, j, k in
        itertools.product(range(Lx), range(Ly), range(Lz))
    }


def vis_contract_around(
    tn,
    tags,
    max_bond,
    cutoff=0.0,
    which='any',
    ndim_sort='max',
    distance_sort='min',
    weight_bonds=False,
    sorter=None,
    fix=None,
    xlims=None,
    ylims=None,
    progbar=True,
    figsize=(5, 5),
    edge_color=(0.85, 0.85, 0.85),
    show_tags=False,
    draw_opts=(),
    **kwargs
):
    import math
    import ipywidgets as widgets
    from autoray import lazy as lz

    tids = tn._get_tids_from_tags(tags, which)
    fix = tn.draw(fix=fix, get='pos')

    span_opts = {
        'ndim_sort': ndim_sort,
        'distance_sort': distance_sort,
        'sorter': sorter,
        'weight_bonds': weight_bonds,
    }

    span = tn.get_tree_span(tids, **span_opts)

    draw_opts = {
        'fix': fix,
        'figsize': figsize,
        'show_tags': show_tags,
        'edge_color': edge_color,
        'xlims': xlims,
        'ylims': ylims,
        'return_fig': True,
        **dict(draw_opts),
    }

    plots = [
        tn._draw_tree_span_tids(
            tids,
            span=span,
            colormap='Blues',
            title='initial-spanning-tree',
            **draw_opts
        )
    ]
    clean()

    def callback_pre_contract(tn, ctids):
        sub_span = [
            (tid1, tid2, d) for tid1, tid2, d in span
            if
            tid1 in tn.tensor_map and
            tid2 in tn.tensor_map
        ]
        s1 = math.log2(tn.tensor_map[ctids[0]].size)
        s2 = math.log2(tn.tensor_map[ctids[1]].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            span=sub_span,
            highlight_tids=ctids,
            colormap='Blues',
            title=f'pre-contract {s1:.2f} {s2:.2f}',
            **draw_opts
        ))
        clean()

    def callback_post_contract(tn, tid):
        sub_span = [
            (tid1, tid2, d) for tid1, tid2, d in span
            if
            tid1 in tn.tensor_map and
            tid2 in tn.tensor_map
        ]
        s1 = math.log2(tn.tensor_map[tid].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            span=sub_span,
            highlight_tids=[tid],
            colormap='Blues',
            title=f'post-contract {s1}',
            **draw_opts
        ))
        clean()

    def callback_pre_compress(tn, tids):
        s1 = math.log2(tn.tensor_map[tids[0]].size)
        s2 = math.log2(tn.tensor_map[tids[1]].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            highlight_tids=[tids[1]],
            highlight_inds_color=(0.0, .6, .3),
            colormap='RdYlBu',
            title=f'pre-compress {s1} {s2}',
            max_distance=kwargs.get('canonize_distance', 0),
            **span_opts, **draw_opts,
        ))
        clean()

    def callback_post_compress(tn, tids):
        s1 = math.log2(tn.tensor_map[tids[0]].size)
        s2 = math.log2(tn.tensor_map[tids[1]].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            highlight_tids=[tids[1]],
            highlight_inds_color=(0.0, .6, .3),
            colormap='RdYlBu',
            title=f'post-compress {s1} {s2}',
            max_distance=kwargs.get('canonize_after_distance', 0),
            **span_opts, **draw_opts
        ))
        clean()

    tnc = tn.copy()

    with lz.shared_intermediates():
        tnc.apply_to_arrays(lz.array)
        Z = tnc.contract_around(
            tags, which,
            max_bond=max_bond,
            cutoff=cutoff,
            span_opts=span_opts,
            callback_pre_contract=callback_pre_contract,
            callback_post_contract=callback_post_contract,
            callback_pre_compress=callback_pre_compress,
            callback_post_compress=callback_post_compress,
            progbar=progbar,
            **kwargs
        )

    def f(x):
        return plots[x]

    widgets.interact(f, x=widgets.IntSlider(0, 0, len(plots) - 1, 1))

    return Z


def vis_contract_compressed(
    tn,
    optimize,
    max_bond,
    cutoff=0.0,
    fix=None,
    pos_update=None,
    xlims=None,
    ylims=None,
    progbar=True,
    figsize=(5, 5),
    edge_color=(0.85, 0.85, 0.85),
    show_tags=False,
    ndim_sort='max',
    distance_sort='min',
    weight_bonds=False,
    sorter=None,
    draw_opts=(),
    title_info=True,
    **kwargs,
):
    import math
    import ipywidgets as widgets
    from autoray import lazy as lz

    fix = tn.draw(fix=fix, get='pos')
    subgraph_sizes = {tid: 1 for tid in fix}

    span_opts = {
        'ndim_sort': ndim_sort,
        'distance_sort': distance_sort,
        'sorter': sorter,
        'weight_bonds': weight_bonds,
    }

    draw_opts = {
        'fix': fix,
        'figsize': figsize,
        'show_tags': show_tags,
        'edge_color': edge_color,
        'xlims': xlims,
        'ylims': ylims,
        'return_fig': True,
        **dict(draw_opts),
    }

    plots = []

    cents = tn.compute_centralities()

    def callback_pre_contract(tn, ctids):
        tid1, tid2 = ctids

        s1 = math.log2(tn.tensor_map[tid1].size)
        s2 = math.log2(tn.tensor_map[tid2].size)
        plots.append(tn.draw(
            highlight_tids=ctids,
            title=f'pre-contract {s1:.2f} {s2:.2f}' if title_info else None,
            **draw_opts
        ))
        clean()
        pos1 = fix.pop(tid1)
        pos2 = fix.pop(tid2)

        sz1 = subgraph_sizes.pop(tid1)
        sz2 = subgraph_sizes.pop(tid2)

        if pos_update == 'inner':
            if cents[tid1] > cents[tid2]:
                fix[tid2] = pos1
            else:
                fix[tid2] = pos2

        elif pos_update == 'subgraph-mean':
            fix[tid2] = (pos1 * sz1 + pos2 * sz2) / (sz1 + sz2)

        else:
            fix[tid2] = pos2

        subgraph_sizes[tid2] = (sz1 + sz2)

    def callback_post_contract(tn, tid):
        s1 = math.log2(tn.tensor_map[tid].size)
        plots.append(tn.draw(
            highlight_tids=[tid],
            title=f'post-contract {s1}' if title_info else None,
            **draw_opts
        ))
        clean()

    def callback_pre_compress(tn, tids):
        s1 = math.log2(tn.tensor_map[tids[0]].size)
        s2 = math.log2(tn.tensor_map[tids[1]].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            highlight_tids=[tids[1]],
            highlight_inds_color=(0.0, .6, .3),
            colormap='RdYlBu',
            title=f'pre-compress {s1} {s2}' if title_info else None,
            max_distance=kwargs.get('canonize_distance', 0),
            **span_opts, **draw_opts,
        ))
        clean()

    def callback_post_compress(tn, tids):
        s1 = math.log2(tn.tensor_map[tids[0]].size)
        s2 = math.log2(tn.tensor_map[tids[1]].size)
        plots.append(tn._draw_tree_span_tids(
            tids,
            highlight_tids=[tids[1]],
            highlight_inds_color=(0.0, .6, .3),
            colormap='RdYlBu',
            title=f'post-compress {s1} {s2}' if title_info else None,
            max_distance=kwargs.get('canonize_after_distance', 0),
            **span_opts, **draw_opts
        ))
        clean()

    tnc = tn.copy()
    with lz.shared_intermediates():
        tnc.apply_to_arrays(lz.array)
        Z = tnc.contract_compressed(
            optimize,
            max_bond=max_bond,
            cutoff=cutoff,
            callback_pre_contract=callback_pre_contract,
            callback_post_contract=callback_post_contract,
            callback_pre_compress=callback_pre_compress,
            callback_post_compress=callback_post_compress,
            progbar=progbar,
            **kwargs,
        )

    def f(x):
        return plots[x]

    widgets.interact(f, x=widgets.IntSlider(0, 0, len(plots) - 1, 1))

    return Z


def vis_contract_compressed_static(
    tn,
    optimize,
    max_bond,
    draw_rotation=0.15,
    draw_squash=1/3,
    ystep_scale=1.0,
    edge_scale=1.0,
    color_compress='#e69f00',
    color_contract='#009e73',
    color_normal='#56b4e9',
    draw_opts=None,
    contract_compressed_opts=None,
    figsize=(8, 8),
):
    import numpy as np
    import matplotlib as mpl
    from math import cos, sin, log2
    from autoray import lazy as lz

    draw_opts = {} if draw_opts is None else dict(draw_opts)
    contract_compressed_opts = ({} if contract_compressed_opts is None else
                                dict(contract_compressed_opts))

    # don't want to actually perform any contractions
    tn = tn.copy()
    tn.apply_to_arrays(lz.array)

    # TN positions, draw normally then rotate and squash into plane
    pos = tn.draw(get='pos', **draw_opts)
    pos = {
        tid: (
            (cos(draw_rotation) * x + sin(draw_rotation) * y),
            (sin(draw_rotation) * x - cos(draw_rotation) * y) * draw_squash,
        )
        for tid, (x, y) in pos.items()
    }

    # keep original positions, dynamically update `pos` for intermediate nodes
    pos0 = pos.copy()

    # keep track of vertical progression
    ystep = ystep_scale / tn.num_tensors
    y0 = max(xy[1] for xy in pos.values()) + ystep
    ymut = [y0]

    lines = []

    def lw(tid, tn):
        return edge_scale * log2(tn.tensor_map[tid].size) / 10 + 1

    def callback_pre_compress(tn, tids):
        tid1, tid2 = tids
        pos1 = pos[tid1]
        pos2 = pos[tid2]
        pos1u = (pos1[0], ymut[0])
        pos2u = (pos2[0], ymut[0])
        size1 = lw(tid1, tn)
        size2 = lw(tid2, tn)
        lines.append({'posA': pos1, 'posB': pos1u, 'linewidth': size1})
        lines.append({'posA': pos2, 'posB': pos2u, 'linewidth': size2})
        lines.append({'posA': pos1u, 'posB': pos2u, 'color': color_compress,
                      'arrowstyle': '<->', 'linewidth': max(size1, size2)})
        pos[tid1] = pos1u
        pos[tid2] = pos2u
        ymut[0] += ystep

    def callback_pre_contract(tn, tids):
        tid1, tid2 = tids
        pos1, pos2 = pos[tid1], pos[tid2]
        pos1u = (pos1[0], ymut[0])
        pos2u = (pos2[0], ymut[0])
        pos12 = ((pos1[0] + pos2[0]) / 2, ymut[0])
        size1 = lw(tid1, tn)
        size2 = lw(tid2, tn)
        lines.append({'posA': pos1, 'posB': pos1u, 'linewidth': size1})
        lines.append({'posA': pos2, 'posB': pos2u, 'linewidth': size2})
        lines.append({'posA': pos1u, 'posB': pos12, 'linewidth': size1,
                      'color': color_contract})
        lines.append({'posA': pos2u, 'posB': pos12, 'linewidth': size2,
                      'color': color_contract})
        pos[tid2] = pos12
        ymut[0] += ystep

    tn.contract_compressed(
        optimize,
        max_bond=max_bond,
        cutoff=0.0,
        callback_pre_compress=callback_pre_compress,
        callback_pre_contract=callback_pre_contract,
        **contract_compressed_opts,
    )

    _, ax = plt.subplots(figsize=figsize)

    tn.draw(
        ax=ax,
        fix=pos0,
        show_tags=False,
        edge_alpha=1.0,
        edge_color='black',
        # node_color='black',
        node_outline_darkness=0.0,
        **draw_opts
    )

    for line in lines:
        line.setdefault('shrinkA', 0)
        line.setdefault('shrinkB', 0)
        line.setdefault('color', color_normal)
        line.setdefault('alpha', 1.0)
        line.setdefault('arrowstyle', '->')
        line.setdefault('mutation_scale', 4)
        ax.add_patch(mpl.patches.FancyArrowPatch(**line))

    ax.set_aspect('equal')
    ysteps = np.arange(y0, ymut[0], ystep)
    ax.set_yticks(ysteps)
    ax.grid(True, axis='y', which='major', linestyle=':')
    ax.set_axisbelow(True)
    ax.set_frame_on(False)

    plt.show()
