from typing import List, Optional  # Imports typing aids for specifying types of variables.

import fire  # Imports the fire library for creating a command-line interface.

from llama import Llama, Dialog  # Imports the Llama class and Dialog type from the llama module.

import json  # Imports the json module for parsing JSON-formatted data.

# Defines the main function with parameters for configuring the text generation process.
def main(
    ckpt_dir: str,  # Path to the directory where model checkpoints are stored.
    tokenizer_path: str,  # Path to the tokenizer used for text encoding/decoding.
    temperature: float = 0.6,  # Controls randomness in text generation, with a default value.
    top_p: float = 0.9,  # Controls diversity of the generated text through top-p sampling, with a default value.
    max_seq_len: int = 512,  # Maximum number of tokens in the input sequence, with a default value.
    max_batch_size: int = 8,  # Maximum number of sequences to process in parallel, with a default value.
    max_gen_len: Optional[int] = None,  # Optional maximum length for generated text. Defaults to model's max if None.
):
    """
    Detailed documentation of the main function explaining its purpose and parameters.
    """

    # Initializes the generator using the specified checkpoint and tokenizer, along with other parameters.
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Opens and reads dialogs from a JSON file named 'dialogs.json'.
    with open('dialogs.json', 'r') as file:
        dialogs: List[Dialog] = json.load(file)

    # Generates text completions for the loaded dialogs using the generator, respecting the provided parameters.
    results = generator.chat_completion(
        dialogs,  # The dialogs to complete.
        max_gen_len=max_gen_len,  # Maximum generation length.
        temperature=temperature,  # Temperature for generation.
        top_p=top_p,  # Top-p value for sampling.
    )

    # Loops through each dialog and its corresponding result, printing them to the console.
    for dialog, result in zip(dialogs, results):
        for msg in dialog:  # Loops through messages in each dialog.
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")  # Prints each message in the dialog.
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"  # Prints the generated response.
        )
        print("\n==================================\n")  # Prints a separator line for readability.

# Checks if this script is executed as the main program and not imported as a module.
if __name__ == "__main__":
    fire.Fire(main)  # Initializes the Fire CLI with the main function, allowing command-line parameters to map to function arguments.
