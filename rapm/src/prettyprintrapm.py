import pandas as pd
import matplotlib.pyplot as plt

from paths import OUTPUTS, RAPM_WITH_PRIOR_ALL, ensure_dirs

ensure_dirs()


def create_pretty_report(input_csv,
                         output_excel=OUTPUTS / 'RAPM_Top_30.xlsx',
                         output_html=OUTPUTS / 'RAPM_Top_30.html',
                         chart_image=OUTPUTS / 'RAPM_Top_30_BarChart.png'):
    """
    Creates a pretty report of the top 30 players by RAPM.
    
    Parameters:
        input_csv (str): Path to the input RAPM CSV file.
        output_excel (str): Path for the output Excel file.
        output_html (str): Path for the output HTML file.
        chart_image (str): Path for saving the bar chart image.
    """
    # Step 1: Read the RAPM CSV
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: The file {input_csv} does not exist.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The file {input_csv} is empty.")
        return
    except Exception as e:
        print(f"An error occurred while reading {input_csv}: {e}")
        return
    df.rename(columns={'RAPM':'SPM','Offensive Prior RAPM':'Off_Prior','Defensive Prior RAPM':'Def_Prior'},inplace=True)
    # Verify necessary columns exist
    required_columns = ['Name', 'Off_Prior', 'Def_Prior', 'SPM','Season']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The input CSV must contain the following columns: {required_columns}")
        return
    
    # Step 2: Select and Sort Top 30 Players
    df_selected = df[['Name', 'Off_Prior', 'Def_Prior', 'SPM','Season']].copy()
    df_sorted = df_selected.sort_values(by='SPM', ascending=False)
    top_30 = df_sorted.head(30).reset_index(drop=True)
    
    # Step 3: Generate a Horizontal Bar Chart
    plt.figure(figsize=(12, 20))  # Width, Height in inches
    plt.barh(top_30['Name'][::-1], top_30['SPM'][::-1], color='skyblue')  # Reverse for descending order
    plt.xlabel('SPM')
    plt.title('Top 30 NBA Players by SPM')
    plt.tight_layout()
    plt.savefig(chart_image)
    plt.close()
    
    # Step 4: Export to Excel with Styling and Embedded Chart
    try:
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            top_30.to_excel(writer, sheet_name='Top 30 RAPM', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Top 30 RAPM']
            
            # Define a format for the header
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1})
            
            # Apply the header format
            for col_num, value in enumerate(top_30.columns.values):
                worksheet.write(0, col_num, value, header_format)
                # Set column widths for better readability
                if value == 'Name':
                    worksheet.set_column(col_num, col_num, 25)
                elif value == 'Season':
                    worksheet.set_column(col_num, col_num, 10)
                elif value == 'SPM':
                    worksheet.set_column(col_num, col_num, 15)
            
            # Insert the bar chart image
            worksheet.insert_image('E2', chart_image, {'x_scale': 0.5, 'y_scale': 0.5})
    except Exception as e:
        print(f"An error occurred while writing to Excel: {e}")
        return
    
    # Step 5: Export to HTML with Styling
    try:
        # Sort again to ensure top SPM is at the top in the HTML
        top_30_sorted = top_30.sort_values(by='SPM', ascending=False)
        html_table = top_30_sorted.to_html(index=False, classes='table table-striped', border=0)
        
        # Define HTML content with basic styling
        html_content = f"""
        <html>
        <head>
            <title>Top 30 NBA Players by SPM</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{background-color: #f5f5f5;}}
                .title {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <h1 class="title">Top 30 NBA Players by SPM</h1>
            {html_table}
        </body>
        </html>
        """
        
        with open(output_html, 'w') as f:
            f.write(html_content)
    except Exception as e:
        print(f"An error occurred while writing to HTML: {e}")
        return
    
    print(f"Top 30 SPM report generated successfully!")
    print(f"- Excel Report: {output_excel}")
    print(f"- HTML Report: {output_html}")
    print(f"- Bar Chart Image: {chart_image}")

if __name__ == "__main__":
    create_pretty_report(RAPM_WITH_PRIOR_ALL)
