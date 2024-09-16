# **ARC-AGI Multimodal Tokenization Plan with Block Multimodal Encoding**

> **Disclaimer**: This is the original plan for the ARC-AGI multimodal  
> tokenization and block encoding mechanism, and it may change as the project  
> evolves.

---

### **Purpose:**

The goal is to tokenize both **text** and **integer grid** (intgrid) data in a  
way that allows the model to process them seamlessly. The tokens will include  
essential metadata in a **token channel encoding** and use specialized metadata  
token channels to handle control information about the **block type** and  
**block shape**.

---

### **Key Considerations:**

- **Sequential Zone Processing**: Zones are processed in order, with each block  
within a zone tokenized and provided with rich metadata to guide the model.  
- **Specialized Modes for Metadata**: In addition to handling content (text or  
grid), specialized **Control Modes** will handle metadata about block type and  
shape.  
- **Unified Token Structure**: All tokenized blocks share a consistent structure,  
using token channel encoding to store mode, zone, sequence, and positional  
information, along with the content.

---

### **Input Data Structure:**

The input is organized into **zones**, where each zone represents a task-related  
content block (e.g., `"rule_statement"`, `"example_grid"`). Each zone contains  
sequenced **blocks**, formatted as:

- **"mode"**: Specifies the data type (e.g., `"text"`, `"intgrid"`).  
- **"payload"**: The actual data, which could be a string for text or a 2D list  
for intgrid data.

#### **Zones and Blocks:**

- **Zones** contain **blocks**, and each block holds a specific type of content.  
The tokenized version of each block includes specialized metadata that provides  
additional control information, such as block type and shape.

---

### **Block Multimodal Encoding:**

In **Block Multimodal Encoding**, each block is converted into a tokenized  
format that includes both content and control-related metadata. This allows the  
model to not only process content but also understand the block's **type** and  
**shape** through separate metadata token channels.

1. **Metadata Token Channels:**  
   - Specialized metadata channels are created to allow the model to generate or  
   process information about the block's shape, type, and other control  
   information.

2. **Mode Slots for Metadata:**  
   - Each specialized mode has its own slot in the **Mode** field of the token  
   channel. These modes are assigned as follows:  
     - **Control Mode [0]:** For handling general control information, such as  
     zone boundaries.  
     - **TextShape Mode [1]:** For embedding information about the shape of text  
     blocks.  
     - **GridShape Mode [2]:** For embedding information about the shape of  
     intgrid blocks.  
     - **Text Mode [3]:** For processing the content of text data.  
     - **Image Mode [4]:** For handling visual data or image-related blocks.

---

### **Token Channel Encoding:**

Each tokenized block (whether text, intgrid, or metadata) is represented by a  
tensor with the following structure:

1. **Mode [1]:**  
   - Specifies the mode of operation. Examples:  
     - `Control Mode [0]`: For general control metadata.  
     - `TextShape Mode [1]`: For information about text block shapes.  
     - `GridShape Mode [2]`: For intgrid block shape metadata.  
     - `Text Mode [3]`: For tokenizing text data.  
     - `Image Mode [4]`: For visual or image data.

2. **Zone [1]:**  
   - Identifies the zone the block belongs to. Zones group task-related content  
   and metadata together.

3. **Sequence [1]:**  
   - Represents the sequence number of the block within its zone, helping the  
   model maintain the correct order.

4. **Data [1]:**  
   - Contains the actual content or control information, depending on the mode.  
     - **For text:** This is a token ID.  
     - **For intgrid:** This is the integer value from the grid.

5. **Shape [D]:**  
   - Represents the overall dimensions of the block being processed. This is  
   used for both text and intgrid modes.

6. **Index [D]:**  
   - Specifies the position within the block (e.g., row and column for  
   intgrids), used to generate positional encodings downstream.

---

### **Tokenization Process:**

#### 1. **Input Parsing and Block Conversion:**  
   - Each zone is processed sequentially, with the blocks in that zone converted  
   into tokenized form.  
   - Specialized modes for metadata (such as shape and control information) are  
   processed separately and embedded into the tokenized sequence.

#### 2. **Text Mode Tokenization:**  
   - **Input Example:** `{"mode": "text", "payload": "Hello, world!"}`  
   - **Processing:**  
     - A text tokenizer converts the string into token IDs.  
     - Metadata (mode, zone, sequence) is added to the tokens.  
     - A **TextShape** metadata token is generated to inform the model about the  
     shape of the text block (e.g., `[200, 0]`).

#### 3. **IntGrid Mode Tokenization:**  
   - **Input Example:** `{"mode": "intgrid", "payload": [[1, 2, 3], [4, 5, 6],  
   [7, 8, 9]]}`  
   - **Processing:**  
     - The grid is flattened into a 1D sequence.  
     - For each grid element, metadata (mode, zone, sequence, shape, index) is  
     added to create the complete token channel encoding.  
     - A **GridShape** metadata token is generated for the grid's shape (e.g.,  
     `[30, 40]`).

#### 4. **Concatenating Tokenized Blocks:**  
   - Once all blocks in a zone are tokenized (along with any control or shape  
   metadata tokens), their outputs are concatenated.  
   - This process is repeated for each zone.
   - Important metadata tokens are placed at the front of each block.
---

### **Specialized Metadata Embedding:**

To ensure the model processes block-level metadata correctly, specialized token  
channels are used for shape and control information. For example:

- **TextShape Mode [1]:** Embeds the shape of a text block into the token  
channel encoding.  
- **GridShape Mode [2]:** Embeds the shape of a grid into the token channel  
encoding.  
- **Control Mode [0]:** Handles control-related information, such as zone  
boundaries and other operational data.

---

### **Decoding Process:**

The **decode** function reconstructs the original data by using the metadata to  
reverse the tokenization process:

- **Text Mode Decoding:** Convert token IDs back into text using a text decoder.  
- **IntGrid Mode Decoding:** Use the shape and index metadata to rebuild the  
original 2D grid from the flattened sequence.  
- **Control and Shape Decoding:** Metadata from specialized modes is decoded  
separately, helping the model interpret block types and shapes.

---

### **Summary of the Approach:**

1. **Unified Tokenization:** Both text and intgrid data are tokenized into a  
unified format, using token channel encoding and specialized metadata channels  
for block shape and control.  
2. **Specialized Metadata Handling:** Control and shape information is embedded  
into the sequence using specialized modes, ensuring that the model can handle  
the structure of the data.  
3. **Sequential Zone Processing:** Each zone is processed in sequence, with  
tokenized content and metadata merged together for efficient model input.
