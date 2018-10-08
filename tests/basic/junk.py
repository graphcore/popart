
        # write the inputs
        print("writing inputs to protobuf files")
        for i in range(len(self.inNames)):
            input_i = self.inputs[i]
            dummy_input_tensor = onnx.numpy_helper.from_array(input_i.numpy())
            input_name_i = self.inNames[i]
            fnIn = os.path.join(dirname, "%s.pb" % (input_name_i, ))
            with open(fnIn, 'wb') as f:
                print("  --writing %s" % (fnIn, ))
                f.write(dummy_input_tensor.SerializeToString())

        # write the outputs
        outputs = self.module(self.inputs)
        print("writing outputs to protobuf files")
        for output, outName in zip(outputs, self.outNames):
            fnOut = os.path.join(dirname, outName + ".pb")
            output_tensor = onnx.numpy_helper.from_array(
                output.detach().numpy())
            with open(fnOut, 'wb') as f:
                print("  --writing %s" % (fnOut, ))
                f.write(output_tensor.SerializeToString())


    #   # we just warm up the model (needed for BN for example)
    #   self.module.train()
    #   for i in range(5):
    #       output = self.module(self.inputs)

    #   # write ONNX model
    #   fnModelWriter0 = os.path.join(dirname, "model0.onnx")
    #   print("writing ONNX model (t=0) to protobuf file")
    #   self.writeOnnxModelWriter(fnModelWriter0)
    #
    #
        # we just warm up the model (needed for BN for example)
        self.module.train()
        for i in range(5):
            output = self.module(self.inputs)






        #set self.outShapes:
        self.outShapes = {}
        for i in range(len(self.outNames)):
            self.outShapes[self.outNames[i]] = outputs[i].shape

        self.module.train()

        # get loss, as per pytrain.py
        # this is what .backward() will be called on
        torchLossTarget = self.getTorchLossTarget()

    def getOutShape(self):
        if self.outShapes is None:
            raise RuntimeError(
                "outShapes not yet set, writeOnnx should be called first")
        else:
            return self.outShapes
