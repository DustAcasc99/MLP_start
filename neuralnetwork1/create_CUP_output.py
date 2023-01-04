#create_csv file for CUP output

import csv

name1, name2, name3 = 'Anna Cinelli', 'Andrea Giorgione', 'Alessio Cascione'
name_string = f"# {name1}\t{name2}\t{name3}\n"
team_name = '# AAA-effect\n'
cup_date = '# ML-CUP22\n'
submission_date = '# 05/01/2023\n'

header_strings = [name_string, team_name, cup_date, submission_date]

output = [[1431212, 14124141], [412412412, 4141241]]


with open("cup_def.csv", 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    for row in header_strings:
        csvfile.write(row)
    for i in range(len(output)):
        print([i+1] + output[i])
        writer.writerow([i+1] + output[i])
