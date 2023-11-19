from pointer_bart import BARTPointer
from transformers import AutoTokenizer

model = BARTPointer.from_pretrained("checkpoints/pointer_bart3/checkpoint-1500")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

input_ids = tokenizer("""(CNN)Following last year's successful U.K. tour, Prince and 3rdEyeGirl are bringing the Hit & Run Tour to the U.S. for the first time. The first -- and so far only -- scheduled show will take place in Louisville, Kentucky, the hometown of 3rdEyeGirl drummer Hannah Welton. Slated for March 14, tickets will go on sale Monday, March 9 at 10 a.m. local time. Prince crowns dual rock charts . A venue has yet to be announced. When the Hit & Run worked its way through the U.K. in 2014, concert venues were revealed via Twitter prior to each show. Portions of the ticket sales will be donated to various Louisville charities. See the original story at Billboard.com. ©2015 Billboard. All Rights Reserved.""", return_tensors="pt").input_ids

model.pointer.pgen_list = []
outputs = model.generate(input_ids=input_ids, max_length=100)
for pgen, token in zip(model.pointer.pgen_list, outputs[0]):
    print(f"{pgen:.2f} {tokenizer.decode(token)}")
    
print()
print("Article:")
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
print()    
print("Summary:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
