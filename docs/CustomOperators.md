## Custom Operator Schemas
* This file is manually created - FFS to figure out how to automate it

* ai.graphcore
  * <a href="#Subsample">Subsample</a>


## ai.graphcore

### <a name="Subsample"></a><a name="subsample">***Subsample***</a>

  The operator subsamples a input tensor by selecting for each dimension every n'th item.
  The sumsample count N is provided for each dimension

#### Version

#### Attributes

#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor to be sub sampled.</dd>
<dt><tt>stride</tt> : list of ints</dt>
<dd>The sub sample count for each dimension. 0 is invalid</dd>
</dl>

#### Output

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor</dd>
</dl>

#### Type Constraints

#### Examples