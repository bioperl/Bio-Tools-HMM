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
     requires => {
                  'perl'          => '5.8.0',
                 },
    );
$build->create_build_script;
