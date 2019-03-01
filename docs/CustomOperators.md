## Custom Operator Schemas
* This file is manually created - FFS to figure out how to automate it

* ai.graphcore
  * <a href="#GroupNormalization">GroupNormalization</a>
  * <a href="#Subsample">Subsample</a>


## ai.graphcore


### <a name="GroupNormalization"></a><a name="groupnormalization">***GroupNormalization***</a>

  The operator groupnormalization applies Group Normalization over a mini-batch of input

#### Version

#### Attributes

<dl>
<dt><tt>num_groups</tt> : int</dt>
<dd>The number of groups to seperate the channels into. The groups are strided (find better explanation).</dd>
<dt><tt>epsilon</tt> : float (default is 1e-05) </dt>
<dd>The epsilon value to use to avoid division by zero.</dd>
</dl>


#### Inputs

<dl>
<dt><tt>input</tt> : T</dt>
<dd>Input tensor to be normalized.</dd>
</dl>

#### Output

<dl>
<dt><tt>output</tt> : T</dt>
<dd>Output tensor</dd>
<dt><tt>mean</tt> : T</dt>
<dd>The mean after the GroupNormalization operator</dd>
<dt><tt>var</tt> : T</dt>
<dd>The variance after the GroupNormalization operator</dd>
</dl>

#### Type Constraints

#### Examples

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