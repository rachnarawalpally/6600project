import pandas as pd

gender = pd.read_csv("gender_clean.csv")

# Let's create a CSV file with about 500 additional female musicians (excluding those already in the uploaded file)
additional_female_musicians = [
    # Sample of female artists across various genres, decades, and regions
    "Aaliyah", "Paula Abdul", "Christina Aguilera", "Fiona Apple", "Tori Amos", "Joan Armatrading", "Melissa Auf der Maur",
    "Avril Lavigne", "Beyoncé", "Mariah Carey", "Cher", "Céline Dion", "Billie Eilish", "Aretha Franklin", "Lady Gaga",
    "Whitney Houston", "Janet Jackson", "Alicia Keys", "Diana Ross", "Taylor Swift", "Shakira", "Annie Lennox", "Patti Smith",
    "Joni Mitchell", "Norah Jones", "Erykah Badu", "Lauryn Hill", "Bjork", "Brandy", "Kesha", "Pink", "Madonna", "Sia",
    "Solange", "Doja Cat", "Saweetie", "Megan Thee Stallion", "Nicki Minaj", "Iggy Azalea", "Janelle Monáe", "M.I.A.",
    "Amy Winehouse", "Florence Welch", "Nina Simone", "Robyn", "JoJo", "Ashanti", "Jessie J", "Tinashe", "Charli XCX",
    "Demi Lovato", "Rihanna", "Tinashe", "Halsey", "Dua Lipa", "Phoebe Bridgers", "Lucy Dacus", "Snail Mail", "Arlo Parks",
    "Olivia Rodrigo", "Kali Uchis", "Victoria Monét", "Jazmine Sullivan", "Summer Walker", "Sabrina Claudio", "Rosalía",
    "Nathy Peluso", "Koffee", "Aya Nakamura", "Angèle", "Zaz", "Lorde", "FKA twigs", "BANKS", "Bat for Lashes", "Lana Del Rey",
    "Natalie Imbruglia", "Nelly Furtado", "Alanis Morissette", "Miley Cyrus", "Tove Lo", "Adele", "Kate Bush", "Tracey Thorn",
    "Debbie Harry", "Courtney Love", "Dolores O'Riordan", "Nico", "PJ Harvey", "Hope Sandoval", "Feist", "Michelle Branch",
    "Vanessa Carlton", "Jewel", "LeAnn Rimes", "Faith Hill", "Kacey Musgraves", "Carrie Underwood", "Reba McEntire",
    "Trisha Yearwood", "Miranda Lambert", "Martina McBride", "Sara Evans", "Shania Twain", "Bonnie Raitt", "Pat Benatar",
    "Stevie Nicks", "Linda Ronstadt", "Joan Jett", "Cyndi Lauper", "Belinda Carlisle", "Susanna Hoffs", "Kim Deal",
    "Corin Tucker", "Kathleen Hanna", "Kim Gordon", "Jenny Lewis", "Sharon Van Etten", "Cat Power", "Julien Baker",
    "St. Vincent", "Ani DiFranco", "Lucinda Williams", "Emmylou Harris", "Brandi Carlile", "Allison Russell",
    "Rhiannon Giddens", "Sandy Denny", "Laura Marling", "Joan Baez", "Judy Collins", "Barbara Streisand", "Linda Perry",
    "Neneh Cherry", "Lisa Lisa", "Coko", "Monica", "Tamar Braxton", "Toni Braxton", "Tamia", "Deborah Cox", "Angie Stone",
    "Kelly Rowland", "Michelle Williams", "Fantasia", "Jennifer Hudson", "Corinne Bailey Rae", "Nao", "India Arie",
    "Yuna", "Utada Hikaru", "BoA", "IU", "CL", "Sunmi", "Heize", "Jessi", "Taeyeon", "HyunA", "Lisa", "Jisoo", "Jennie", "Rosé",
] 

# Create a DataFrame
new_female_df = pd.DataFrame({'name': additional_female_musicians, 'gender': ['female'] * len(additional_female_musicians)})

# Save to CSV
output_path = "artist_genders_more_women.csv"
new_female_df.to_csv(output_path, index=False)

output_path

## join to existing CSV

# Concatenate the new data with the existing data
combined_gender_df = pd.concat([gender, new_female_df], ignore_index=True)

# Remove duplicates if any
combined_gender_df = combined_gender_df.drop_duplicates(subset='name', keep='first')

# Save the updated data back to a CSV file
combined_output_path = "artist_genders_more_women.csv"
combined_gender_df.to_csv(combined_output_path, index=False)

combined_output_path