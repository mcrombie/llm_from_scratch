import urllib.request, re
url = (
    "https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt"
) 
file_path = "the-verdict.txt" 
urllib.request.urlretrieve(url, file_path)



def main():
    with open("the-verdict.txt", "r", encoding="utf-8") as f: 
        raw_text = f.read() 
    print("Total number of character:", len(raw_text)) 
    print(raw_text[:99])
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text) 
    preprocessed = [item.strip() for item in preprocessed if item.strip()] 
    print(len(preprocessed))
    print(preprocessed[:30])

if __name__ == "__main__":
    main()
