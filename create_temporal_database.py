import pandas as pd
import geopandas as gpd
from Toolbox.file_manage import reverse_pid, convert_pid

txt_path = "merged.txt"
shp_path = r"mars_mro_crism_mtrdr_c0a\mars_mro_crism_mtrdr_c0a.shp"
output_dir = "CRISM_Metadata_Database"

target_cols = ['ProductId','CenterLat','CenterLon','MaxLat','MinLat','EastLon','WestLon',
               'EmAngle','InAngle','PhAngle','SolLong','UTCstart']

if __name__ == "__main__":
   # Step 1: Read the TXT file and process it into a DataFrame
    lines = []
    with open(txt_path, 'r') as file:
        for line in file:
            ROIid, base_img, warp_img_lst = line.strip().split()
            lst = [ROIid, reverse_pid(base_img)] + [reverse_pid(warp_img) for warp_img in warp_img_lst.split(',')]
            lines.append(lst)
    
    # Convert to DataFrame
    group_data = {line[0]: line[1:] for line in lines}  # First column as group ID, rest as IDs
    df_groups = pd.DataFrame.from_dict(group_data, orient='index').transpose()
    df_groups = df_groups.melt(var_name='GroupID', value_name='ProductId').dropna()
    
    # Step 2: Read the shapefile with GeoPandas
    gdf = gpd.read_file(shp_path, columns=target_cols)[target_cols]
    
    # Merge the attributes from the shapefile to df_groups based on 'ProductId'
    merged_df = df_groups.merge(gdf, on='ProductId', how='left')
    #merged_df.drop(columns=['geometry'], inplace=True)
    # Step 3: Re-arrange each group dataframe by 'UTCstart'
    if 'UTCstart' in merged_df.columns:
        merged_df = merged_df.sort_values(by='UTCstart')
    merged_df.drop_duplicates(keep='first', inplace=True)
    
    # Step 4: Save each group as a separate CSV file
    lines = []
    for group_id, group_df in merged_df.groupby('GroupID'):
        group_df.drop(columns=['GroupID'], inplace=True)
        product_ids = [convert_pid(pid) for pid in group_df['ProductId'].values.tolist()]
        group_df['ProductId'] = product_ids
        group_df.to_csv(f"{output_dir}/{group_id}.csv", index=False)
        
        frt_ids = [pid for pid in product_ids if pid.startswith("frt")]
        if frt_ids:
            other_ids = [pid for pid in product_ids if not pid.startswith("frt")]  # Handle different prefixes
            formatted_line = f"{group_id} {frt_ids[0]} " + ",".join(frt_ids[1:] + other_ids)
        else:
            formatted_line = f"{group_id} {product_ids[0]} " + ",".join(product_ids[1:])
        lines.append(formatted_line)
    
    with open(output_dir+".txt", 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + "\n")