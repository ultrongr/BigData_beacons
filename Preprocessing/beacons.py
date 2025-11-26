import pandas as pd
import time


input_file = "../Raw Data/beacons_dataset_commas.csv"
output_file = "../Preprocessed Data/beacons_dataset_commas.csv"




def drop_invalid_part_ids(_df):

    unique_part_ids = _df['part_id'].unique()

    def is_numeric(value):
        if pd.isna(value):
            return True
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    non_numeric_ids = [
        part_id for part_id in unique_part_ids if not is_numeric(part_id)
    ]


    for _id in non_numeric_ids:
        _df.loc[_df['part_id'] == _id, 'count'] = _df.loc[_df['part_id'] == _id].shape[0]
        _df = _df.drop(_df.index[_df['part_id'] == _id])

    return _df

def print_unique_part_ids(_df):
    unique_part_ids = _df['part_id'].unique()

    unique_part_ids.sort()
    print(f"Total unique participant IDs: {len(unique_part_ids)}")
    for part_id in unique_part_ids:
        print(f"{part_id}: {_df.loc[_df['part_id'] == part_id].shape[0]}")

def fix_room_names(_df):

    _df = _df.dropna(subset=['room'])

    aliases = dict()

    # --- Kitchen Variants ---
    aliases['Kitchen'] = [
        "kitchen", "Kitcen", "Kithen", "Kitvhen",
        "Kichen", "Kiychen", "Kitcheb"
    ]
    aliases['Kitchen2'] = ["kitchen2", "Kitchen2"]

    # --- Living Room Variants ---
    aliases['Livingroom'] = [
        "Leavingroom", "LivingRoom", "Sittingroom", "Sittigroom",
        "SittingOver", "LeavingRoom", "SeatingRoom", "LuvingRoom",
        "Livingroom1", "Liningroom", "Leavivinroom", "livingroom",
        "Living", "Livingroon", "LivibgRoom", "Luvingroom1", "Sittinroom",
        "SittingRoom", "Sitingroom"
    ]
    aliases['Livingroom2'] = ["livingroom2", "LivingRoom2", "Livingroom2"]

    # --- Bedroom Variants ---
    aliases['Bedroom'] = [
        "bedroom", "Bedroom1", "Bedroom1st",
        "Bedroom-1", "bedroom", "Bedroom1st"
    ]
    aliases['Bedroom2'] = ["Bedroom2", "2ndRoom"]

    # --- Bathroom Variants ---
    aliases['Bathroom'] = [
        "Bathroon", "Baghroom", "Bsthroom", "Bathroim",
        "Bqthroom", "Bathroom1", "Bathroom-1"
    ]

    # --- Dining / Diner Room ---
    aliases['DiningRoom'] = [
        "DiningRoom", "DinerRoom", "Dinerroom", "DinnerRoom",
        "DinningRoom"
    ]

    # --- Office Variants ---
    aliases['Office'] = [
        "Office", "Office1", "Office1st", "Workroom"
    ]
    aliases['Office2'] = ["Office2", "Office-2"]

    # --- Hall / Entrance ---
    aliases['Hall'] = ["Hall", "Entrance", "ExitHall"]

    # --- Outdoor & Garden ---
    aliases['Outdoor'] = ["Outdoor", "Garden", "Veranda"]

    # --- Storage / Misc ---
    aliases['Storage'] = ["Storage"]
    aliases['Garage'] = ["Garage"]
    aliases['Laundry'] = ["Laundry", "LaundryRoom"]
    aliases['Library'] = ["Library"]
    aliases['Pantry'] = ["Pantry"]
    aliases['Desk'] = ["Desk"]
    aliases['Box'] = ["Box", "Box-1"]
    aliases['Two'] = ["Two"]
    aliases['Three'] = ["Three", "three"]
    aliases['Four'] = ["Four"]
    aliases['TV'] = ["TV"]
    aliases['nan'] = ["nan"]

    drop_names = [
        "TV", "Two", "Three", "One", "Four", "K", "T", "Guard"
    ]

    for standard_name, variant_list in aliases.items():
        for variant in variant_list:
            _df.loc[_df['room'] == variant, 'room'] = standard_name
    
    for drop_name in drop_names:
        _df = _df.drop(_df.index[_df['room'] == drop_name])

    return _df

def print_unique_rooms(_df):
    unique_rooms = _df['room'].unique()

    unique_rooms.sort()
    print(f"Total unique rooms: {len(unique_rooms)}")
    for room in unique_rooms:
        print(f"{room}: {_df.loc[_df['room'] == room].shape[0]}")

# def fix_continuity_issues(_df):
    


    


if __name__ == "__main__":

    time1 = time.time()
    df = pd.read_csv("../Raw Data/beacons_dataset_commas.csv")
    print(f"Time taken to read the CSV: {time.time() - time1} seconds")

    df = drop_invalid_part_ids(df)
    print_unique_part_ids(df)
    
    df = fix_room_names(df)
    print_unique_rooms(df)
