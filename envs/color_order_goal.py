"""Pure goal semantics shared by collection, simulation and evaluation."""
import math
import random

TASK_NAME = "blocks_ranking_color_order"
ORDERS = ("RGB", "RBG", "GRB", "GBR", "BRG", "BGR")
COLORS = {"R": "red", "G": "green", "B": "blue"}


def resolve_order(order, seed):
    if order == "random":
        # Never consume the scene's NumPy RNG or the language sampler's RNG.
        return random.Random(int(seed)).choice(ORDERS)
    if order not in ORDERS:
        raise ValueError(f"target_order must be one of {ORDERS} or 'random', got {order!r}")
    return order


def collection_order(order, episode_index, *, balanced):
    if order == "random" and balanced:
        return ORDERS[episode_index % len(ORDERS)]
    if order != "random":
        resolve_order(order, 0)
    return order


def episode_info(order, scene_seed):
    order = resolve_order(order, scene_seed)
    return {
        "info": {f"{{{slot}}}": f"{COLORS[color]} block" for slot, color in zip("ABC", order)},
        "task_goal": {
            "order_id": order,
            "target_order": [COLORS[color] for color in order],
            "scene_seed": int(scene_seed),
            "reference_frame": "world_x_increasing",
        },
    }


def validate_episode_info(info):
    goal = info.get('task_goal', {})
    if goal.get('order_id') not in ORDERS or 'scene_seed' not in goal:
        raise ValueError('Color-order episodes require an explicit task_goal')
    expected = episode_info(goal['order_id'], goal['scene_seed'])
    if info.get('info') != expected['info'] or goal != expected['task_goal']:
        raise ValueError('Instruction parameters disagree with the color-order goal')


def row_success(positions, order, half_size, table_z, grippers_open=True):
    resolve_order(order, 0)
    points = [positions[color] for color in order]
    if not grippers_open or not all(math.isfinite(float(v)) for p in points for v in p):
        return False
    gaps = [points[i + 1][0] - points[i][0] for i in range(2)]
    return bool(
        all(2 * half_size + 0.003 <= gap <= 0.13 for gap in gaps)
        and max(p[1] for p in points) - min(p[1] for p in points) <= 0.03
        and all(abs(p[2] - (table_z + half_size)) <= 0.012 for p in points)
    )


def summarize_results(records):
    by_order = {}
    for order in ORDERS:
        items = [r for r in records if r['order_id'] == order]
        successes = sum(bool(r['success']) for r in items)
        by_order[order] = {
            "episodes": len(items), "successes": successes,
            "success_rate": successes / len(items) if items else None,
        }
    rates = [v['success_rate'] for v in by_order.values()]
    return {
        "by_order": by_order,
        "macro_success_rate": sum(rates) / 6 if all(r is not None for r in rates) else None,
    }
