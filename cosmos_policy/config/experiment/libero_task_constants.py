"""Task-id mappings for LIBERO-Cosmos-Policy suites.

Supports both naming styles:
- success_only folders: ``libero_goal_regen`` (etc.)
- all_episodes filenames: ``libero_goal`` (etc.)
"""

from __future__ import annotations


LIBERO_SUITE_TASK_ID_TO_DESCRIPTION: dict[str, dict[int, str]] = {
    "libero_spatial": {
        0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
        2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
        3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
        4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
        5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
        6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
        7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
        8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
        9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
    },
    "libero_object": {
        0: "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        2: "pick_up_the_salad_dressing_and_place_it_in_the_basket",
        3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
        4: "pick_up_the_ketchup_and_place_it_in_the_basket",
        5: "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
        6: "pick_up_the_butter_and_place_it_in_the_basket",
        7: "pick_up_the_milk_and_place_it_in_the_basket",
        8: "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
        9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
    },
    "libero_goal": {
        0: "open_the_middle_drawer_of_the_cabinet",
        1: "put_the_bowl_on_the_stove",
        2: "put_the_wine_bottle_on_top_of_the_cabinet",
        3: "open_the_top_drawer_and_put_the_bowl_inside",
        4: "put_the_bowl_on_top_of_the_cabinet",
        5: "push_the_plate_to_the_front_of_the_stove",
        6: "put_the_cream_cheese_in_the_bowl",
        7: "turn_on_the_stove",
        8: "put_the_bowl_on_the_plate",
        9: "put_the_wine_bottle_on_the_rack",
    },
    "libero_10": {
        0: "put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        1: "put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
        2: "turn_on_the_stove_and_put_the_moka_pot_on_it",
        3: "put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
        4: "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
        5: "pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
        6: "put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
        7: "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
        8: "put_both_moka_pots_on_the_stove",
        9: "put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
    },
}


def task_description_to_replay_name(task_description: str) -> str:
    return task_description.replace("_", " ")


def get_replay_tasks(suite_name: str, task_ids: list[int]) -> list[str]:
    id_to_task = LIBERO_SUITE_TASK_ID_TO_DESCRIPTION[suite_name]
    return [task_description_to_replay_name(id_to_task[task_id]) for task_id in task_ids]

