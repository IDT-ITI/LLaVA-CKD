<h2>LLaVA dataset<a class="headerlink" href="#id1" title="Link to this heading"></a></h2>
<ul>
<li><p><strong>Pretraining Images</strong>: The pretraining images of LLaVA is from the 558K subset of the LAION-CC-SBU dataset. Download as follows.</p>
<ul class="simple">
<li><p>LAION-CC-SBU-558K: <a class="reference external" href="https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip">images.zip</a>.</p></li>
</ul>
</li>
<li><p><strong>Pretraining Annotations</strong>: The pretraining annotations of LLaVA. Download as follows.</p>
<ul class="simple">
<li><p>pretraining annotations: <a class="reference external" href="https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain">blip_laion_cc_sbu_558k.json</a>.</p></li>
</ul>
</li>
<li><p><strong>SFT Images</strong>: The SFT images of LLaVA. Download as follows.</p>
<ul class="simple">
<li><p>LAION-CC-SBU-558K: Already download as “LAION-CC-SBU-558K” in Pretraining Images.</p></li>
<li><p>COCO: This dataset is from the <a class="reference external" href="https://cocodataset.org/">COCO2017_challenge</a>. Download: <a class="reference external" href="http://images.cocodataset.org/zips/train2017.zip">train2017</a>.</p></li>
</ul>
<ul class="simple">
<li><p>GQA: <a class="reference external" href="https://cs.stanford.edu/people/dorarad/gqa/about.html">GQA_project_page</a>. Download: <a class="reference external" href="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip">gqa_images</a>.</p></li>
</ul>
<ul class="simple">
<li><p>OCR-VQA: <a class="reference external" href="https://ocr-vqa.github.io/">OCR-VQA_project_page</a>. Download: <a class="reference external" href="https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_">download_script</a>. We save all files as <code class="docutils literal notranslate"><span class="pre">.jpg</span></code>.</p></li>
</ul>
<ul class="simple">
<li><p>TextVQA: <a class="reference external" href="https://textvqa.org/">TextVQA_project_page</a>. Download: <a class="reference external" href="https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip">trainval_images</a>.</p></li>
</ul>
<ul class="simple">
<li><p>VisualGenome: <a class="reference external" href="https://homes.cs.washington.edu/~ranjay/visualgenome/index.html">VisualGenome_project_page</a>. Download: <a class="reference external" href="https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip">part1</a>, <a class="reference external" href="https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip">part2</a>.</p></li>
</ul>
</li>
<li><p><strong>SFT Annotations</strong>: The SFT annotations of LLaVA. Download as follows.</p>
<ul class="simple">
<li><p>SFT annotations: <a class="reference external" href="https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json">llava_v1_5_mix665k.json</a>.</p></li>
</ul>
</li>
</ul>
</section>
<section id="id2">
<h2>ShareGPT4V dataset<a class="headerlink" href="#id2" title="Link to this heading"></a></h2>
<ul>
<li><p><strong>Pretraining and SFT Images</strong>: The images of ShareGPT4V. Download as follows.</p>
<ul class="simple">
<li><p>LAION-CC-SBU-558K: Already download as “LAION-CC-SBU-558K” in LLaVA’s Pretraining Images.</p></li>
<li><p>COCO: Already download as “COCO” in LLaVA’s SFT Images.</p></li>
<li><p>WebData &amp; Share_TextVQA: This dataset is curated by the <a class="reference external" href="https://github.com/InternLM/InternLM-XComposer/tree/main/projects/ShareGPT4V">ShareGPT4V_project</a>. Download: <a class="reference external" href="https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax">images</a>. Only for academic usage.</p></li>
</ul>
<ul class="simple">
<li><p>SAM: This dataset is collected by Meta. Download: <a class="reference external" href="https://ai.meta.com/datasets/segment-anything-downloads/">sam_images</a>. We only use 000000~000050.tar for now. If you just want to use ShareGPT4V for SFT, you can quickly download <a class="reference external" href="https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link">9K_images</a>.</p></li>
</ul>
<ul class="simple">
<li><p>GQA: Already download as “GQA” in LLaVA’s SFT Images.</p></li>
<li><p>OCR-VQA: Already download as “OCR-VQA” in LLaVA’s SFT Images.</p></li>
<li><p>TextVQA: Already download as “TextVQA” in LLaVA’s SFT Images.</p></li>
<li><p>VisualGenome: Already download as “VisualGenome” in LLaVA’s SFT Images.</p></li>
</ul>
</li>
