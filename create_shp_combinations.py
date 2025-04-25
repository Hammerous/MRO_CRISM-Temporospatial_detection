import Toolbox.shp_process as shp
import Toolbox.graph_process as gph
import pandas as pd
import os

shp_file = r"InAngle65_intsct.shp"
cols = ['ProductId', 'geometry']

if __name__ == "__main__":
    gdf = shp.open_shp(shp_file, cols)
    pairlst = shp.find_relation_lst(gdf)
    print(f"Basic Pairs: {len(pairlst)}")
    df = {'ProductId': [], 'Max View': [], 'Group Id': []}
    for size, sub_lst in gph.intsec2graph(pairlst, 3):
        # Flatten sub_lst (which is a list of lists)
        flattened_product_ids = [item for sublist in sub_lst for item in sublist]
        original_indices = [i for i, sublist in enumerate(sub_lst) for _ in sublist]
        df['ProductId'].extend(flattened_product_ids)
        df['Max View'].extend([size] * len(flattened_product_ids))
        df['Group Id'].extend(original_indices)
        print(f"Size {size}: {len(sub_lst)}")
    
    df = pd.DataFrame(df)
    df.to_csv(f'{os.path.basename(shp_file).split(".")[0]}.csv', index=False)