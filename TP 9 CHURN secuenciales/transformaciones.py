import pandas as pd

def apply_transformations_complete(df):
    df_transformed = df.copy()
    
    transformations = {
        1: {"new_name": "status-cuenta", "mapping": {"A14": 2, "A11": 1, "A12": 3, "A13": 4}},
        
        2: {
            "new_name": "duracion-meses",
            "transform_func": lambda x: (
                1 if pd.notna(x) and x <= 10 else
                2 if pd.notna(x) and x <= 20 else
                3 if pd.notna(x) and x <= 30 else
                4 if pd.notna(x) and x <= 40 else 5
            )
        },
        
        3: {"new_name": "credit-history", "mapping": {"A30": 1, "A31": 2, "A32": 3, "A33": 4, "A34": 5}},
        
        4: {
            "new_name": "credit-purpose", 
            "mapping": {"A49": 1, "A48": 2, "A47": 3, "A46": 4, "A45": 5, 
                        "A44": 6, "A43": 7, "A42": 8, "A41": 9, "A40": 10, "A410": 11}
        },
        
        5: {"new_name": "credit-amount", "no_transform": True},
        
        6: {"new_name": "saving-account-amount", "mapping": {"A65": 1, "A61": 2, "A62": 3, "A63": 2, "A64": 1}},
        
        7: {"new_name": "antiguedad-trabajo", "mapping": {"A75": 1, "A74": 2, "A73": 3, "A72": 4, "A71": 5}},
        
        8: {"new_name": "tasa-interes", "no_transform": True},
        
        9: {"new_name": "estado-civil", "mapping": {"A91": 1, "A92": 2, "A93": 3, "A94": 4, "A95": 5}},
        
        10: {"new_name": "garante", "mapping": {"A101": 3, "A102": 2, "A103": 1}},
        
        11: {"new_name": "11", "no_transform": True},
        
        12: {"new_name": "propiedades", "mapping": {"A124": 4, "A123": 3, "A122": 2, "A121": 1}},
        
        13: {
            "new_name": "edad",
            "transform_func": lambda x: 1 if pd.notna(x) and x < 30 else 1
        },
        
        14: {"new_name": "14", "mapping": {"A141": 141, "A142": 142, "A143": 143}},
        
        15: {"new_name": "alojamiento", "mapping": {"A153": 1, "A151": 2, "A152": 3}},
        
        16: {"new_name": "cantidad-creditos", "no_transform": True},
        
        17: {"new_name": "trabajo", "mapping": {"A171": 4, "A172": 3, "A173": 2, "A174": 1}},
        
        18: {"new_name": "cantidad-manutencion", "no_transform": True},
        
        19: {"new_name": "telefono", "mapping": {"A191": 1, "A192": 0}},
        
        20: {"new_name": "trabajo-domestico", "mapping": {"A201": 0, "A202": 1}},
    }
    
    for col_num, transform_info in transformations.items():
        if col_num in df_transformed.columns:
            new_name = transform_info["new_name"]
            
            if "mapping" in transform_info:
                df_transformed[col_num] = df_transformed[col_num].map(transform_info["mapping"])
            elif "transform_func" in transform_info:
                if new_name in ["duracion-meses", "edad"]:
                    df_transformed[col_num] = pd.to_numeric(df_transformed[col_num], errors='coerce')
                df_transformed[col_num] = df_transformed[col_num].apply(transform_info["transform_func"])
            
            df_transformed = df_transformed.rename(columns={col_num: new_name})
    
    if "Rechazo" in df_transformed.columns:
        df_transformed["Rechazo"] = df_transformed["Rechazo"].map({1: 0, 2: 1})
    
    columns_to_drop_final = [col for col in ["11", "14"] if col in df_transformed.columns]
    if columns_to_drop_final:
        df_transformed = df_transformed.drop(columns=columns_to_drop_final)
    
    return df_transformed

# Load the Excel file
# Assuming 'Base_Clientes Alemanes.xlsx' is in the same directory as the script
df = pd.read_excel('Base_Clientes Alemanes.xlsx')

# Apply the transformations
df_transformed_complete = apply_transformations_complete(df)

# Save the transformed DataFrame to a new Excel file
output_file_name = 'Base_Clientes Alemanes Transformed.xlsx'
df_transformed_complete.to_excel(output_file_name, index=False)
print(f"Transformed data successfully saved to '{output_file_name}'")