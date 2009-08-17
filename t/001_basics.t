## Test framework for HMM XS stuff
## $Id$
use strict;
use warnings;

BEGIN {
    use Bio::Root::Test;
    test_begin(-tests => 40);
    use_ok('Bio::Matrix::Scoring');
    use_ok('Bio::Tools::HMM'); 
}

my $debug = test_debug();

sub debugging {
    return unless $debug;
    print STDERR $_[0] if @_ == 1;
    printf STDERR ($_[0], $_[1]) if @_ == 2;
}

my $hmm = Bio::Tools::HMM->new('-symbols' => "123456", '-states' => "FL");

my ($seq1, $obs1);
$seq1 =  "315116246446644245311321631164152133625144543631656626566666";
$obs1 =  "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLL";
$seq1 .= "651166453132651245636664631636663162326455236266666625151631";
$obs1 .= "LLLLLLFFFFFFFFFFFFLLLLLLLLLLLLLLLLFFFLLLLLLLLLLLLLLFFFFFFFFF";
$seq1 .= "222555441666566563564324364131513465146353411126414626253356";
$obs1 .= "FFFFFFFFLLLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq1 .= "366163666466232534413661661163252562462255265252265435353336";
$obs1 .= "LLLLLLLLFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq1 .= "233121625364414432335163243633665562466662632666612355245242";
$obs1 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFF";

my ($seq2, $obs2);
$seq2 =  "544552213525245666363632432522253566166546666666533666543261";
$obs2 =  "FFFFFFFFFFFFLLLLLLLLLLLFFFFFFFFFFFFLLLLLLLLLLLLLLLLLLLFFFFFF";
$seq2 .= "363546253252546524422555242223224344432423341365415551632161";
$obs2 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";
$seq2 .= "144212242323456563652263346116214136666156616666566421456123";
$obs2 .= "FFFFFFLLLFFFFFFFFFFFFFFFFFFFFFFFFLFLLLLLLLLLLLLLLLLFFFFFFFFF";
$seq2 .= "346313546514332164351242356166641344615135266642261112465663";
$obs2 .= "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF";

my @seqs = ($seq1, $seq2);

debugging "Baum-Welch Training\n";
debugging "===================\n";

$hmm->baum_welch_training(\@seqs);

debugging "Initial Probability Array:\n";
my $init = $hmm->init_prob;

is(scalar(@$init), 2);
my @test_vals = (0.499992,0.500008);

foreach my $s (@{$init}) {
    float_is($s, shift @test_vals, "Initial prob.");
    debugging("%g\t", $s);
}
debugging "\n";

debugging "Transition Probability Matrix:\n";
my $matrix = $hmm->transition_prob;

@test_vals = (0.499992,0.500008,0.499992,0.500008,
              0.499992,0.500008,0.499992,0.500008);
foreach my $r ($matrix->row_names) {
    foreach my $c ($matrix->column_names) {
        float_is($matrix->entry($r, $c), shift @test_vals, "Initial prob.");
        debugging "%g\t", $matrix->entry($r, $c);
    }
    debugging "\n";
}

@test_vals = (0.133333,0.143333,0.163333,0.123333,0.143333,0.293333,
              0.133333,0.143333,0.163333,0.123333,0.143333,0.293333,              
              );
debugging "Emission Probability Matrix:\n";
$matrix = $hmm->emission_prob;
foreach my $r ($matrix->row_names) {
    foreach my $c ($matrix->column_names) {
        float_is($matrix->entry($r, $c), shift @test_vals, "Emission prob.");
        debugging "%g\t", $matrix->entry($r, $c);
    }
    debugging "\n";
}

debugging "\n";
debugging "Log Probability of sequence 1: %g\n", $hmm->likelihood($seq1);
debugging "Log Probability of sequence 2: %g\n", $hmm->likelihood($seq2);
debugging "\n";
debugging "Statistical Training\n";
debugging "====================\n";

my @obs = ($obs1, $obs2);
$hmm->statistical_training(\@seqs, \@obs);

debugging "Initial Probability Array:\n";
$init = $hmm->init_prob;
$hmm->init_prob($init);
@test_vals = (1,0);
foreach my $s (@{$init}) {
    float_is($s, shift @test_vals, "Initial prob.");
    debugging "%g\t", $s;
}
debugging "\n";

@test_vals = (0.970732,0.0292683,
              0.0638298,0.93617);
debugging "Transition Probability Matrix:\n";
$matrix = $hmm->transition_prob;
$hmm->transition_prob($matrix);
foreach my $r ($matrix->row_names) {
    foreach my $c ($matrix->column_names) {
        float_is($matrix->entry($r, $c), shift @test_vals, "Transition prob.");
        debugging "%g\t", $matrix->entry($r, $c);
    }
    debugging "\n";
}

@test_vals = (0.160194 ,0.174757 ,0.179612,0.160194 ,0.160194,0.165049,
              0.0744681,0.0744681,0.12766 ,0.0425532,0.106383,0.574468);
debugging "Emission Probability Matrix:\n";
$matrix = $hmm->emission_prob;
$hmm->emission_prob($matrix);
foreach my $r ($matrix->row_names) {
    #print STDERR join(',', map {sprintf("%g",$matrix->entry($r, $_))} $matrix->column_names)."\n";
    foreach my $c ($matrix->column_names) {
        float_is($matrix->entry($r, $c), shift @test_vals, "Transition prob.");
        debugging "%g\t", $matrix->entry($r, $c);
    }
    debugging "\n";
}

my $expected =
    'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'.
    'FFFFFLLLLLLLLLLLLLLLLLLLLLFFFFFFFFFFFFLL'.
    'LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLFFFFFFFF'.
    'FFFFFFFFFLLLLLLLLLLLFFFFFFFFFFFFFFFFFFFF'.
    'FFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLLFFFFFFFF'.
    'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF'.
    'FFFFFFFFFFFFFFFFFFFFFFFFFFFLLLLLLLLLLLLL'.
    'LLLLLLLLLFFFFFFFFFFF';
debugging "Viterbi Algorithm:\n";
my $obs3 = $hmm->viterbi($seq1);
is($obs3, $expected);
debugging "%s\n", $obs3;

