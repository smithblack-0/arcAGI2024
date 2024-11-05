import model

model.build_recurrent_decoder_v1(100, 10, 10, 10, 512,

                                 0.2, 0.1, 0.01,
                                 "FastLinearMemory",
                                 deep_memory_details={"d_address": 4,
                                                      "d_memory": 20,
                                                      "num_memories": 100,
                                                      "num_read_heads" : 10,
                                                      "num_write_heads" : 10,
                                                      },
                                 layer_controller_variant="PseudoMarkovBankSelector",
                                 layer_controller_details={}
                                 )
