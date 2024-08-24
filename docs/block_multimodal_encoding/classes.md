# Schemas and Distributions.

## SchemaTracker

**Purpose**

* Keep track of:
  * the various schemas that exist
  * their string mode names
  * their numeric representation

**Dependencies**

* num_dimensions: The number of supported target dims.
* logit_slots: The number of allowed

**Design**

Internally, we store data in two fields

* A names field, which is a list of strings
* A schema field, which is a (N x num_dimensions) tensor containing the schema, where
  "N" is the different schemas.

Additionally, each specified schema has an id associated with it, and that 
id is the same id as it's name in the list (or schema tensor.)

### Method: register_schema

**Purpose**

* Registers a schema.
* Optional: Performs conversion if provided schema is not long enough.

**Accepts**

* name [str]: The name to call the schema. Must be unique.
* schema: The schema.

**Design**

Most of the work on this method goes into validating. With regards to the string:

* Check if the name is already in use, if it is throw an error

Then, for the schema:

* Must be a list of ints >= 0.
* If not long enough we pad to num_dims with zeros
* If too long, we throw a "schemas with dimensions in excess of {num_dim} not supported" error.
* The sum of the schema must be less than or equal to the number of logit slots

Once these are passed, we store the schema inside the class. I would lean towards
storing it as a tuple inside a dictionary mapped to the schema name myself. 

Note that we must declare and associate a schema_id int when registering the schema. The
id may never be reused, will start at zero, and count up.

### Method: get_schema_id

**Purpose**

The purpose of this method is to get the schema_id that was assigned to a schema.
This is needed when preprocessing data to ensure the data ends up assigned to the
right schema.

**Accepts**

schema_name [str]: The name of the schema

**Returns**

schema_id [int]: The schema id it has been associated with. An int

**Raises**

key_error: If the schema was never registered.

**Design**

Fairly straightforward. We return the index of the name in the list.

### Method: fetch_schemas

**Purpose**

The purpose of this method is to fetch a tensor of schemas based on a tensor
of schema_ids. 

**Accepts**

schema_ids [torch.Tensor]:
    * An int tensor where each integer represents a schema we wish to reference.
    * Shape (...)

**Returns**

schemas [torch.]:
    * Another int tensor specifying the schemas
    * Shape (... x num_dimensions)

**Design**

We use vector indexing for this. The schema_ids tensor is used
to select from the schema tensor along the N dimension.

## ModeConverter

**Purpose**

The mode manager is responsible for abstracting away everything
that has to do with managing a particular mode as would be 
viewed by the generative models. 

This includes but is not solely limited to 

## PipelineTracker

**Purpose**

The model tracker is the adapter that keeps track of the various
machine learning models that are available to handle the project, and
associate models with datatypes





## Class: DataConverter

**Purpose**

The data converter is responsible for being able to convert to, or
convert from, a particular mode. It is thus responsible for both
encoding and decoding.

**Dependencies**

* SchemaTracker: So we can discover what schema is associated with a particular mode

### Method: encode

**Purpose**

Encodes an input case

### Method: decode



## function: make_separation_mask

**Purpose**

Several classes will need to be able to make separation masks in order to function. We
make a careful, generalized function to perform this task.

**Accepts**

* schemas_tensor:
  * Represents the active 
  * Shape (... x num_dimensions)

* num_dims

## class: CrossEntropy

**Purpose**

Performs cross entropy loss. Ignores unneeded dimensions. Incorporates label
smoothing cleanly.
