color_options = ['red','blue','purple','brown','green']

res_path = '../Tracking'

csv_file_ext = "*.csv"

npy_save_path = '../npy_files'

wells_to_genetype_dict = {
  **dict.fromkeys(['D2','D3','D4','G5','G6','G7'], "control"),
  **dict.fromkeys(['D5','D6','D7'], "Grb2"),
  **dict.fromkeys(['F2','F3','F4'], "Gab1"),
  **dict.fromkeys(['G2','G3','G4'], "MET+Gab1"),
  **dict.fromkeys(['E5','E6',"E7"], "MET+Grb2")
}
