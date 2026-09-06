# Run with: bundle exec ruby _checks/math_rendering.rb
# Render real articles with and without front matter defaults. A running
# Jekyll server may still have the configuration from before math was enabled.
require 'jekyll'
require 'nokogiri'

source = File.expand_path('..', __dir__)
checked = 0
[{}, { 'defaults' => [] }].each do |overrides|
  site = Jekyll::Site.new(Jekyll.configuration({ 'source' => source }.merge(overrides)))
  site.reset
  site.read
  site.generate

  verify = lambda do |document, expected|
    output = Jekyll::Renderer.new(site, document).run
    html = Nokogiri::HTML(output)
    loaders = html.css('script[src]').select { |script| script['src'].include?('MathJax.js') }
    configs = html.css('script[type="text/x-mathjax-config"]')
    unless loaders.size == expected && configs.size == expected
      abort "Math loader regression in #{document.relative_path}: expected #{expected} loader and config, got #{loaders.size} and #{configs.size}"
    end
    if expected == 1
      abort "Math config must precede loader in #{document.relative_path}" unless output.index('text/x-mathjax-config') < output.index('MathJax.js')
      abort "Inline math configuration missing in #{document.relative_path}" unless configs.first.text.include?("['$','$']")
    end
    output
  end

  sample_output = nil
  site.posts.docs.each do |post|
    output = verify.call(post, 1)
    sample_output = output if post.basename_without_ext.include?('SignedDualAttention')
    checked += 1
  end

  output = sample_output || abort('Signed Dual Attention sample missing')
  abort 'Inline TeX lost during Markdown conversion' unless output.include?('$\tanh$')
  abort 'Display TeX lost during Markdown conversion' unless output.include?('\[\begin{align}')

  page = site.pages.find { |candidate| candidate.url == '/about_me/' }
  verify.call(page, 0)
  page.data['math'] = true
  verify.call(page, 1)
end
puts "Passed: #{checked} post renders, inline/display TeX preservation, and optional math on standard pages."
