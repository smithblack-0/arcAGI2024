

## Class: ReadTransformFeedback:

### Premise

* We need to be able to read the transform feedback
* This can be done with attention

### Accepts

* Query: Shape (batch x num_queries x cognition_dim)
* ConstructionCommands: Shape (batch x num_executions x instruction_order x transforms_dim)
* TransformFeedback: Shape (batch x num_executions x instruction_order x transforms_dim)
* Mask: Shape (batch x num_executions x instruction_order x transforms_dim)

### Returns

* response: (batch x num_queries x cognition_dim)

### Design

We use attention, of course. It is just a very clever version of it. We treat this as finding
and incorporating information from an address. We want to find 

* First: The execution case to read from
* Second: The instructions we really care about

After this, we can put these parts together and do a weighted sum over two dimensions.

**Execution focus**

We 

## Class: MakeTransformDirective

### Premise

* We need to be able to make transform direcr