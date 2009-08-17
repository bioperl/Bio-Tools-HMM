use Module::Build;
my $build = Module::Build->new
    (
     module_name => 'Bio::Tools::HMM',
     dist_version => '1.6.1',
     dist_name    => 'Bio-Tools-HMM',
     dist_abstract => 'perform Hidden Markov Model calculations',
     license  => 'perl',
     create_makefile_pl => 'passthrough',
     recursive_test_files => 1,
     create_readme => 1,
     create_license => 1,
     meta_merge => {
                    resources => {
                                  repository => 'svn://code.open-bio.org/bioperl/Bio-Tools-HMM'
                                 },
                   },
     build_requires => {
                        'ExtUtils::CBuilder' => '0',
                        ($^O =~ /^(dos|MSWin32)$/ ? ('ExtUtils::Install' => 1.39) : ()),
                       },
     requires => {
                  'perl'                 => '5.8.0',
                  'Bio::Root::Version'   => '1.006009',
                  # just need Version until we split things up (upon where each dist gets it's own version)
                  #'Bio::Matrix::Scoring' => '1.6.0',
                 },
     c_source => 'src',
     #xs_files       => { 'HMM.xs' => 'lib/Bio/Ext/HMM.xs' },
     extra_compiler_flags => ['-lm'],

    );
$build->create_build_script;
