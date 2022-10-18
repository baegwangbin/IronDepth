# IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty

Official implementation of the paper

> **IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty** \
> BMVC 2022 \
> [Gwangbin Bae](https://baegwangbin.com), [Ignas Budvytis](https://mi.eng.cam.ac.uk/~ib255/), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2210.03676) [[demo]](https://www.youtube.com/watch?v=mf8keH9brF0) [[project page]](https://baegwangbin.github.io/IronDepth/)

<p align="center">
  <img width=70% src="https://github.com/baegwangbin/IronDepth/blob/main/docs/img/irondepth/IronDepth_short.gif">
</p>

## Summary

* We use surface normal to propagate depth between pixels.
* We formulate depth refinement/upsampling as classification of choosing the neighboring pixel to propagate from.
* Our method can be used as a post-processing tool to improve the accuracy of the existing depth estimation methods.
* Our method can seamlessly be applied to depth completion. Sparse depth measurements to can be propagated to the neighboring pixels, improving the accuracy of the overall prediction.


## Citation

If you find our work useful in your research please consider citing our paper:

```
@InProceedings{Bae2022,
    title   = {IronDepth: Iterative Refinement of Single-View Depth using Surface Normal and its Uncertainty}
    author  = {Gwangbin Bae and Ignas Budvytis and Roberto Cipolla},
    booktitle = {British Machine Vision Conference (BMVC)},
    year = {2022}                         
}
```

